"""
Aspect Extraction Pipeline (Phase 4).
Extracts fine-grained customer complaint aspects from Amazon customer reviews
using zero-shot Natural Language Inference (NLI) multi-label classification.

Supports:
- Domain-specific aspect taxonomies (Electronics vs. Fashion vs. Common)
- Multi-label aspect detection with confidence scoring
- Comprehensive multi-label evaluation metrics (Precision, Recall, F1)
- Aspect frequency aggregation among negative reviews
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Taxonomies as specified in requirements
COMMON_ASPECTS: List[str] = [
    "packaging",
    "quality",
    "price",
    "delivery",
    "customer_service",
]

ELECTRONICS_SPECIFIC: List[str] = [
    "battery",
    "performance",
    "compatibility",
    "durability",
]

FASHION_SPECIFIC: List[str] = [
    "size_fit",
    "material",
    "design",
    "color",
    "durability",
]

DOMAIN_ASPECTS: Dict[str, List[str]] = {
    "Electronics": COMMON_ASPECTS + ELECTRONICS_SPECIFIC,
    "Fashion": COMMON_ASPECTS + FASHION_SPECIFIC,
}

# Verbalized label mapping for high-precision NLI hypothesis formulation
ASPECT_VERBALIZATIONS: Dict[str, str] = {
    "battery": "battery or charging",
    "performance": "performance or functionality",
    "compatibility": "compatibility or connection",
    "durability": "durability or breaking",
    "size_fit": "size and fit",
    "material": "material or fabric",
    "design": "design or style",
    "color": "color",
    "packaging": "packaging or box",
    "quality": "product quality",
    "price": "price or value",
    "delivery": "delivery or shipping",
    "customer_service": "customer service or return",
}

# Inverted verbalization mapping
VERBALIZATION_TO_ASPECT: Dict[str, str] = {v: k for k, v in ASPECT_VERBALIZATIONS.items()}

DEFAULT_MODEL_NAME = "valhalla/distilbart-mnli-12-3"
DEFAULT_THRESHOLD = 0.40


class AspectExtractor:
    """
    Zero-Shot NLI Aspect Extractor for customer complaints.
    Maps customer review texts to multi-label aspect taxonomies.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        default_threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.default_threshold = default_threshold

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1

        logger.info(
            "Initializing Zero-Shot AspectExtractor with '%s' on %s...",
            self.model_name,
            "CUDA" if self.device == 0 else "CPU",
        )
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )

    def get_candidate_aspects(self, domain: str) -> List[str]:
        """Retrieve valid taxonomy aspects for a given domain."""
        clean_domain = domain.capitalize() if domain else "Electronics"
        if clean_domain not in DOMAIN_ASPECTS:
            # Fallback to union if domain is generic
            return list(set(DOMAIN_ASPECTS["Electronics"] + DOMAIN_ASPECTS["Fashion"]))
        return DOMAIN_ASPECTS[clean_domain]

    def extract_aspects(
        self,
        text: str,
        domain: str = "Electronics",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extract aspect complaints from a single customer review.

        Returns:
            Dict containing:
                - review: input text
                - domain: product domain
                - detected_aspects: list of identified aspect labels
                - confidence: dict of {aspect: probability score}
        """
        th = threshold if threshold is not None else self.default_threshold
        candidates = self.get_candidate_aspects(domain)
        verbalized_candidates = [ASPECT_VERBALIZATIONS.get(a, a) for a in candidates]

        res = self.classifier(
            text,
            candidate_labels=verbalized_candidates,
            hypothesis_template="This review complains about {}.",
            multi_label=True,
        )

        detected_aspects = []
        confidence_dict = {}

        for verbalized_label, score in zip(res["labels"], res["scores"]):
            aspect_key = VERBALIZATION_TO_ASPECT.get(verbalized_label, verbalized_label)
            confidence_dict[aspect_key] = round(float(score), 4)
            if score >= th:
                detected_aspects.append(aspect_key)

        return {
            "review": text,
            "domain": domain,
            "detected_aspects": detected_aspects,
            "confidence": confidence_dict,
        }

    def batch_extract(
        self,
        texts: List[str],
        domains: List[str],
        threshold: Optional[float] = None,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Batch extraction of aspects across multiple customer reviews.
        """
        results = []
        total = len(texts)
        for i, (text, domain) in enumerate(zip(texts, domains)):
            if (i + 1) % 25 == 0 or (i + 1) == total:
                logger.info("Processed %d / %d reviews for aspect extraction...", i + 1, total)
            res = self.extract_aspects(text, domain=domain, threshold=threshold)
            results.append(res)
        return results


def parse_aspect_string(aspects_str: Any) -> List[str]:
    """Parse comma, semicolon, or list formatted aspect strings into a clean list."""
    if aspects_str is None:
        return []
    if isinstance(aspects_str, (list, tuple, np.ndarray)):
        return [str(a).strip().lower() for a in aspects_str if str(a).strip()]
    if pd.isna(aspects_str):
        return []

    # Split by semicolon, comma, or pipe
    cleaned = str(aspects_str).replace(";", ",").replace("|", ",")
    items = [a.strip().lower() for a in cleaned.split(",") if a.strip()]
    return items


def evaluate_aspect_extraction(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    target_aspects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate multi-label aspect predictions against ground-truth annotations.
    Calculates per-aspect and global (Micro, Macro) Precision, Recall, and F1.
    """
    if target_aspects is None:
        all_aspects = sorted(list(set(a for row in (y_true + y_pred) for a in row)))
    else:
        all_aspects = target_aspects

    aspect_to_idx = {a: i for i, a in enumerate(all_aspects)}
    n_samples = len(y_true)
    n_aspects = len(all_aspects)

    # Multi-label binary indicator matrices
    Y_true_bin = np.zeros((n_samples, n_aspects), dtype=int)
    Y_pred_bin = np.zeros((n_samples, n_aspects), dtype=int)

    for i, labels in enumerate(y_true):
        for label in labels:
            if label in aspect_to_idx:
                Y_true_bin[i, aspect_to_idx[label]] = 1

    for i, labels in enumerate(y_pred):
        for label in labels:
            if label in aspect_to_idx:
                Y_pred_bin[i, aspect_to_idx[label]] = 1

    # Overall Micro and Macro metrics
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        Y_true_bin, Y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        Y_true_bin, Y_pred_bin, average="macro", zero_division=0
    )

    # Per-aspect metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        Y_true_bin, Y_pred_bin, average=None, zero_division=0
    )

    per_aspect_dict = {}
    for i, aspect in enumerate(all_aspects):
        per_aspect_dict[aspect] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i]),
        }

    return {
        "sample_count": n_samples,
        "aspect_count": n_aspects,
        "micro_precision": round(float(p_micro), 4),
        "micro_recall": round(float(r_micro), 4),
        "micro_f1": round(float(f1_micro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_aspect": per_aspect_dict,
    }


def compute_aspect_frequencies(
    df: pd.DataFrame,
    aspects_column: str = "aspect_labels",
    domain_column: Optional[str] = "domain",
) -> pd.DataFrame:
    """
    Calculate the occurrence count and percentage of each aspect among negative reviews.
    """
    records = []
    domains = [None] if domain_column not in df.columns else [None] + list(df[domain_column].unique())

    for dom in domains:
        subset = df if dom is None else df[df[domain_column] == dom]
        total_reviews = len(subset)
        if total_reviews == 0:
            continue

        counts: Dict[str, int] = {}
        for aspects in subset[aspects_column]:
            parsed = parse_aspect_string(aspects)
            for a in set(parsed):
                counts[a] = counts.get(a, 0) + 1

        for aspect, count in counts.items():
            records.append({
                "domain": dom if dom else "All",
                "aspect": aspect,
                "count": count,
                "total_reviews": total_reviews,
                "frequency_pct": round((count / total_reviews) * 100.0, 2),
            })

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df = res_df.sort_values(by=["domain", "count"], ascending=[True, False]).reset_index(drop=True)
    return res_df
