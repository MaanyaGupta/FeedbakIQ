"""
Sentiment Analysis Pipeline (Phase 2): TF-IDF + Logistic Regression Baseline.
Trains and evaluates baseline models on Electronics, Fashion, and Combined reviews.
Supports reproducible train/val/test splitting, stratified sampling, comprehensive
metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report),
with special analysis of class imbalance and the neutral class.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.config import (
        PROCESSED_REVIEWS_FILE,
        MODELS_DIR,
        RANDOM_STATE,
    )
    from src.preprocessing import clean_text
except ImportError:
    from config import (
        PROCESSED_REVIEWS_FILE,
        MODELS_DIR,
        RANDOM_STATE,
    )
    from preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["negative", "neutral", "positive"]


def load_sentiment_data(
    data_path: Optional[Path] = None,
    domain: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load processed reviews dataset.
    Optionally filter by domain ('Electronics', 'Fashion', or None for Combined).
    """
    target_path = data_path or PROCESSED_REVIEWS_FILE
    if not target_path.exists():
        raise FileNotFoundError(
            f"Processed reviews file not found at '{target_path}'. "
            "Please run metadata and preprocessing pipeline first."
        )

    try:
        df = pd.read_parquet(target_path)
    except Exception:
        df = pd.read_parquet(target_path, engine="fastparquet")
    logger.info("Loaded %d reviews from '%s'.", len(df), target_path.name)

    if domain and domain.lower() != "combined":
        df = df[df["domain"].str.lower() == domain.lower()].copy().reset_index(drop=True)
        logger.info("Filtered to domain '%s': %d reviews remaining.", domain, len(df))

    return df


def prepare_text_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare text features and sentiment targets.
    Combines 'title' and 'text' to provide strong sentiment signals.
    """
    if "sentiment" not in df.columns:
        raise ValueError("Missing 'sentiment' column in DataFrame.")

    # Combine title and text with punctuation separator
    titles = df["title"].fillna("").astype(str).str.strip()
    texts = df["text"].fillna("").astype(str).str.strip()
    combined_text = titles.where(titles == "", titles + ". ") + texts

    # Light cleaning preserving case and negations
    cleaned_features = combined_text.apply(lambda t: clean_text(t) if pd.notna(t) else "")
    targets = df["sentiment"].astype(str).str.strip().str.lower()

    return cleaned_features, targets


def split_dataset(
    X: pd.Series,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Split dataset into train, validation, and test sets using stratified sampling.
    Default split: 70% Train, 15% Validation, 15% Test.
    """
    temp_size = test_size + val_size
    # First split: Train vs (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: Validation vs Test (split temp 50/50 if test_size == val_size)
    relative_test_ratio = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        "Split dataset: Train=%d, Val=%d, Test=%d (Total=%d).",
        len(X_train),
        len(X_val),
        len(X_test),
        len(X),
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def build_baseline_pipeline(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with TF-IDF Vectorizer and Logistic Regression.
    Uses sublinear term frequency scaling and balanced class weighting.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
    )

    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    return pipeline


def evaluate_sentiment_model(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    split_name: str = "Test",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained pipeline on a dataset split.
    Calculates Accuracy, Precision, Recall, F1 (weighted and macro),
    Confusion Matrix, Classification Report, and Neutral Class diagnostics.
    """
    classes = class_names or CLASS_NAMES
    preds = pipeline.predict(X)

    acc = float(accuracy_score(y, preds))
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y, preds, average="weighted", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y, preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(y, preds, labels=classes)
    report_dict = classification_report(
        y, preds, labels=classes, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y, preds, labels=classes, zero_division=0
    )

    # Detailed neutral class extraction
    neutral_stats = report_dict.get("neutral", {})
    neutral_precision = float(neutral_stats.get("precision", 0.0))
    neutral_recall = float(neutral_stats.get("recall", 0.0))
    neutral_f1 = float(neutral_stats.get("f1-score", 0.0))
    neutral_support = int(neutral_stats.get("support", 0))

    results = {
        "split": split_name,
        "sample_count": len(y),
        "accuracy": acc,
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "neutral_precision": neutral_precision,
        "neutral_recall": neutral_recall,
        "neutral_f1": neutral_f1,
        "neutral_support": neutral_support,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }

    return results


def save_trained_model(
    pipeline: Pipeline,
    domain: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save trained baseline model pipeline to models directory."""
    target_dir = output_dir or MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"baseline_logistic_{domain.lower()}.joblib"
    save_path = target_dir / filename
    joblib.dump(pipeline, save_path)
    logger.info("Saved trained '%s' baseline model to: %s", domain, save_path)
    return save_path


def train_and_evaluate_domain(
    domain: str,
    df_reviews: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Execute full training and evaluation workflow for a domain:
    1. Filter domain data (or use combined).
    2. Extract text features and sentiment targets.
    3. Perform stratified 70/15/15 train/val/test split.
    4. Train TF-IDF + LogisticRegression baseline model.
    5. Evaluate on Validation and Test sets.
    6. Save model artifact under models/.
    """
    logger.info("=== Starting Sentiment Baseline for Domain: '%s' ===", domain)

    if domain.lower() != "combined":
        df_domain = df_reviews[df_reviews["domain"].str.lower() == domain.lower()].copy().reset_index(drop=True)
    else:
        df_domain = df_reviews.copy().reset_index(drop=True)

    if len(df_domain) == 0:
        raise ValueError(f"No records found for domain '{domain}' in dataset.")

    X, y = prepare_text_features(df_domain)
    splits = split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=random_state)

    # Train model
    pipeline = build_baseline_pipeline(
        ngram_range=(1, 2),
        max_features=5000,
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
    )
    pipeline.fit(splits["X_train"], splits["y_train"])

    # Evaluate on Validation set
    val_results = evaluate_sentiment_model(
        pipeline, splits["X_val"], splits["y_val"], split_name="Validation", class_names=CLASS_NAMES
    )

    # Evaluate on Test set
    test_results = evaluate_sentiment_model(
        pipeline, splits["X_test"], splits["y_test"], split_name="Test", class_names=CLASS_NAMES
    )

    # Save model
    saved_path = None
    if save_model:
        saved_path = save_trained_model(pipeline, domain=domain)

    domain_summary = {
        "domain": domain,
        "total_samples": len(df_domain),
        "train_samples": len(splits["X_train"]),
        "val_samples": len(splits["X_val"]),
        "test_samples": len(splits["X_test"]),
        "val_metrics": val_results,
        "test_metrics": test_results,
        "model_path": str(saved_path) if saved_path else None,
    }

    return domain_summary


def print_evaluation_summary(domain_results: Dict[str, Dict[str, Any]]) -> None:
    """Print detailed classification reports, confusion matrices, and comparison table."""
    for domain, res in domain_results.items():
        test_m = res["test_metrics"]
        print("\n" + "=" * 65)
        print(f"DOMAIN: {domain.upper()} (Test Set: {test_m['sample_count']} samples)")
        print("=" * 65)
        print(f"Overall Accuracy:   {test_m['accuracy']:.4f}")
        print(f"Weighted F1-Score:  {test_m['weighted_f1']:.4f}  |  Weighted Precision: {test_m['weighted_precision']:.4f}  |  Weighted Recall: {test_m['weighted_recall']:.4f}")
        print(f"Macro F1-Score:     {test_m['macro_f1']:.4f}  |  Macro Precision:    {test_m['macro_precision']:.4f}  |  Macro Recall:    {test_m['macro_recall']:.4f}")

        print("\nClassification Report (Test Set):")
        print(test_m["classification_report_text"])

        print("Confusion Matrix (Labels: ['negative', 'neutral', 'positive']):")
        cm = np.array(test_m["confusion_matrix"])
        print(f"  Pred Negative  Pred Neutral  Pred Positive")
        for i, row in enumerate(cm):
            lbl = CLASS_NAMES[i].capitalize()
            print(f"Actual {lbl:<8}: {row[0]:<14} {row[1]:<13} {row[2]:<13}")

        print("\nNeutral Class Deep-Dive:")
        print(f"- Support (True Neutrals): {test_m['neutral_support']}")
        print(f"- Precision:               {test_m['neutral_precision']:.4f}")
        print(f"- Recall:                  {test_m['neutral_recall']:.4f}")
        print(f"- F1-Score:                {test_m['neutral_f1']:.4f}")

    # Comparison Table
    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON TABLE (TEST SET)")
    print("=" * 65)
    print(f"| {'Domain':<12} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} |")
    print(f"| {':-----------':<12} | {'-------:':>8} | {'---------:':>9} | {'-------:':>8} | {'---:':>8} |")
    for domain, res in domain_results.items():
        t = res["test_metrics"]
        print(f"| {domain:<12} | {t['accuracy']:>8.4f} | {t['weighted_precision']:>9.4f} | {t['weighted_recall']:>8.4f} | {t['weighted_f1']:>8.4f} |")


def run_sentiment_pipeline(
    data_path: Optional[Path] = None,
    save_models: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run end-to-end sentiment baseline pipeline across:
    1. Electronics
    2. Fashion
    3. Combined dataset
    """
    df_reviews = load_sentiment_data(data_path)
    domains = ["Electronics", "Fashion", "Combined"]
    domain_results: Dict[str, Dict[str, Any]] = {}

    for domain in domains:
        res = train_and_evaluate_domain(
            domain=domain,
            df_reviews=df_reviews,
            random_state=RANDOM_STATE,
            save_model=save_models,
        )
        domain_results[domain] = res

    print_evaluation_summary(domain_results)

    # Save evaluation summary to JSON
    summary_json_path = MODELS_DIR / "baseline_evaluation_metrics.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        # Filter for JSON-serializable fields
        exportable = {}
        for d, data in domain_results.items():
            exportable[d] = {
                "total_samples": data["total_samples"],
                "train_samples": data["train_samples"],
                "val_samples": data["val_samples"],
                "test_samples": data["test_samples"],
                "test_accuracy": data["test_metrics"]["accuracy"],
                "test_weighted_f1": data["test_metrics"]["weighted_f1"],
                "test_weighted_precision": data["test_metrics"]["weighted_precision"],
                "test_weighted_recall": data["test_metrics"]["weighted_recall"],
                "test_macro_f1": data["test_metrics"]["macro_f1"],
                "test_neutral_f1": data["test_metrics"]["neutral_f1"],
                "model_path": data["model_path"],
            }
        json.dump(exportable, f, indent=2)
    logger.info("Saved baseline evaluation metrics summary to: %s", summary_json_path)

    return domain_results


if __name__ == "__main__":
    run_sentiment_pipeline()
