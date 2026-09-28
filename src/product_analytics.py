"""
Product-Level Customer Feedback Analytics Layer (Phase 4 Extension).
Aggregates review sentiments, rating distributions, and fine-grained aspect complaints
at the product level (parent_asin) across Electronics and Fashion domains.

Provides:
- Reusable APIs:
    * get_product_sentiment(product_id)
    * get_product_aspects(product_id)
    * get_product_summary(product_id)
- Persistent product insights table stored at data/processed/product_insights.parquet
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_PRODUCTS_FILE,
    PROCESSED_REVIEWS_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRODUCT_INSIGHTS_FILE = PROCESSED_DATA_DIR / "product_insights.parquet"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Human-readable aspect naming for clean customer-facing summaries
ASPECT_DISPLAY_NAMES = {
    "battery": "Battery & Charging",
    "performance": "Performance & Functionality",
    "compatibility": "Compatibility & Connection",
    "durability": "Durability & Build Quality",
    "size_fit": "Size & Fit",
    "material": "Material & Fabric",
    "design": "Design & Style",
    "color": "Color & Appearance",
    "packaging": "Packaging & Box Condition",
    "quality": "General Product Quality",
    "price": "Price & Value",
    "delivery": "Delivery & Shipping",
    "customer_service": "Customer Service & Returns",
}


def read_parquet_safe(file_path: Path) -> pd.DataFrame:
    """Read parquet with fastparquet fallback."""
    try:
        return pd.read_parquet(file_path)
    except Exception:
        return pd.read_parquet(file_path, engine="fastparquet")


def predict_review_sentiments(
    texts: List[str],
    model_name: str = SENTIMENT_MODEL_NAME,
    batch_size: int = 32,
) -> Tuple[List[str], List[float]]:
    """Predict 3-class sentiment (negative, neutral, positive) using top RoBERTa model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Generating sentiment predictions for %d reviews using %s...", len(texts), model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    all_preds = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            pred_indices = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)

        all_preds.extend([id2label[idx] for idx in pred_indices])
        all_confs.extend([round(float(c), 4) for c in confs])

    return all_preds, all_confs


def build_product_insights(
    reviews_file: Optional[Path] = None,
    products_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build and save the comprehensive product-level customer feedback analytics table.
    Aggregates review counts, sentiment proportions, average ratings, and top complaint aspects.
    """
    rf = reviews_file or PROCESSED_REVIEWS_FILE
    pf = products_file or PROCESSED_PRODUCTS_FILE
    out_path = output_file or PRODUCT_INSIGHTS_FILE

    logger.info("Loading reviews from '%s' and products from '%s'...", rf.name, pf.name)
    df_reviews = read_parquet_safe(rf)
    df_products = read_parquet_safe(pf)

    # Prepare combined text for sentiment inference and aspect extraction
    titles = df_reviews["title"].fillna("").astype(str).str.strip()
    texts = df_reviews["text"].fillna("").astype(str).str.strip()
    combined_texts = titles.where(titles == "", titles + ". ") + texts
    df_reviews["full_text"] = combined_texts

    # Predict sentiments using top RoBERTa model
    predicted_sentiments, predicted_confs = predict_review_sentiments(combined_texts.tolist())
    df_reviews["predicted_sentiment"] = predicted_sentiments
    df_reviews["predicted_confidence"] = predicted_confs
    df_reviews.to_parquet(PROCESSED_DATA_DIR / "reviews_with_predictions.parquet", index=False)

    # Initialize AspectExtractor for negative reviews
    logger.info("Extracting complaint aspects for negative reviews...")
    from src.aspect_extractor import AspectExtractor
    aspect_extractor = AspectExtractor(default_threshold=0.35)

    # Reviews considered negative if rating-derived negative OR model-predicted negative
    is_negative = (df_reviews["sentiment"] == "negative") | (df_reviews["predicted_sentiment"] == "negative")
    neg_indices = df_reviews[is_negative].index

    aspect_results = {}
    for idx in neg_indices:
        row = df_reviews.loc[idx]
        ext_res = aspect_extractor.extract_aspects(row["full_text"], domain=row["domain"])
        aspect_results[idx] = ext_res["detected_aspects"]

    df_reviews["extracted_aspects"] = [aspect_results.get(i, []) for i in df_reviews.index]

    # Aggregate by product (parent_asin)
    product_rows = []
    unique_products = df_reviews["parent_asin"].unique()

    for parent_asin in unique_products:
        prod_reviews = df_reviews[df_reviews["parent_asin"] == parent_asin]
        total_revs = len(prod_reviews)
        if total_revs == 0:
            continue

        domain = prod_reviews["domain"].iloc[0]

        # Match product metadata
        meta_match = df_products[df_products["parent_asin"] == parent_asin]
        if not meta_match.empty:
            prod_title = str(meta_match["title"].iloc[0])
            price = meta_match["price"].iloc[0] if "price" in meta_match.columns else None
            meta_avg_rating = meta_match["average_rating"].iloc[0] if "average_rating" in meta_match.columns else None
        else:
            prod_title = f"Product {parent_asin}"
            price = None
            meta_avg_rating = None

        # Sentiment counts based on ground-truth / rating-derived sentiment
        pos_cnt = int((prod_reviews["sentiment"] == "positive").sum())
        neu_cnt = int((prod_reviews["sentiment"] == "neutral").sum())
        neg_cnt = int((prod_reviews["sentiment"] == "negative").sum())

        pos_pct = round((pos_cnt / total_revs) * 100.0, 1)
        neu_pct = round((neu_cnt / total_revs) * 100.0, 1)
        neg_pct = round((neg_cnt / total_revs) * 100.0, 1)

        # Model predicted sentiment counts
        pred_pos_cnt = int((prod_reviews["predicted_sentiment"] == "positive").sum())
        pred_neu_cnt = int((prod_reviews["predicted_sentiment"] == "neutral").sum())
        pred_neg_cnt = int((prod_reviews["predicted_sentiment"] == "negative").sum())

        pred_pos_pct = round((pred_pos_cnt / total_revs) * 100.0, 1)
        pred_neu_pct = round((pred_neu_cnt / total_revs) * 100.0, 1)
        pred_neg_pct = round((pred_neg_cnt / total_revs) * 100.0, 1)

        # Average rating from reviews
        avg_rating = round(float(prod_reviews["rating"].mean()), 2)

        # Calculate top complaint aspects among negative reviews
        neg_prod_reviews = prod_reviews[prod_reviews["sentiment"] == "negative"]
        total_negatives = len(neg_prod_reviews)

        aspect_counts: Dict[str, int] = {}
        for aspects in neg_prod_reviews["extracted_aspects"]:
            for a in set(aspects):
                aspect_counts[a] = aspect_counts.get(a, 0) + 1

        top_aspects = []
        for a, count in sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True):
            pct = round((count / total_negatives) * 100.0, 1) if total_negatives > 0 else 0.0
            top_aspects.append({
                "aspect": a,
                "display_name": ASPECT_DISPLAY_NAMES.get(a, a.replace("_", " ").title()),
                "count": int(count),
                "percentage": pct,
            })

        # Top complaint formatted summary string
        if top_aspects:
            top_complaints_text = "; ".join(
                [f"{ta['display_name']} ({ta['percentage']}%)" for ta in top_aspects[:4]]
            )
        else:
            top_complaints_text = "No severe complaint clusters detected."

        product_rows.append({
            "parent_asin": parent_asin,
            "domain": domain,
            "title": prod_title,
            "price": price,
            "average_rating": avg_rating,
            "meta_average_rating": meta_avg_rating,
            "total_reviews": total_revs,
            "positive_reviews": pos_cnt,
            "neutral_reviews": neu_cnt,
            "negative_reviews": neg_cnt,
            "positive_percentage": pos_pct,
            "neutral_percentage": neu_pct,
            "negative_percentage": neg_pct,
            "pred_positive_percentage": pred_pos_pct,
            "pred_neutral_percentage": pred_neu_pct,
            "pred_negative_percentage": pred_neg_pct,
            "top_complaints_summary": top_complaints_text,
            "top_complaint_aspects_json": json.dumps(top_aspects),
        })

    df_insights = pd.DataFrame(product_rows)
    df_insights = df_insights.sort_values(by=["domain", "total_reviews"], ascending=[True, False]).reset_index(drop=True)

    # Save to parquet
    df_insights.to_parquet(out_path, index=False)
    logger.info("Saved product insights table to '%s' (%d products).", out_path, len(df_insights))

    return df_insights


class ProductAnalytics:
    """
    Analytics querying class powering downstream chatbot and reporting queries.
    Caches product insights in memory for sub-millisecond retrieval.
    """

    def __init__(self, insights_file: Optional[Path] = None):
        self.insights_file = insights_file or PRODUCT_INSIGHTS_FILE
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        if not self.insights_file.exists():
            logger.info("Product insights file not found at %s. Building it now...", self.insights_file)
            self._df = build_product_insights(output_file=self.insights_file)
        else:
            self._df = read_parquet_safe(self.insights_file)

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            self._load()
        return self._df

    def find_product(self, product_id: str) -> Optional[pd.Series]:
        """Find product row by parent_asin or title substring (case-insensitive)."""
        pid = str(product_id).strip().upper()
        # Direct parent_asin match
        match = self.data[self.data["parent_asin"].str.upper() == pid]
        if not match.empty:
            return match.iloc[0]

        # Substring title match
        title_match = self.data[self.data["title"].str.lower().str.contains(str(product_id).lower(), na=False)]
        if not title_match.empty:
            return title_match.iloc[0]

        return None


# Global singleton instance for easy import
_analytics_engine = ProductAnalytics()


def get_product_sentiment(product_id: str) -> Dict[str, Any]:
    """
    Retrieve sentiment metrics for a specific product.
    Returns:
        Dict with total_reviews, positive_reviews, neutral_reviews, negative_reviews,
        positive_percentage, neutral_percentage, negative_percentage, average_rating.
    """
    row = _analytics_engine.find_product(product_id)
    if row is None:
        raise ValueError(f"Product '{product_id}' not found in product insights database.")

    return {
        "parent_asin": row["parent_asin"],
        "domain": row["domain"],
        "title": row["title"],
        "total_reviews": int(row["total_reviews"]),
        "positive_reviews": int(row["positive_reviews"]),
        "neutral_reviews": int(row["neutral_reviews"]),
        "negative_reviews": int(row["negative_reviews"]),
        "positive_percentage": float(row["positive_percentage"]),
        "neutral_percentage": float(row["neutral_percentage"]),
        "negative_percentage": float(row["negative_percentage"]),
        "average_rating": float(row["average_rating"]),
    }


def get_product_aspects(product_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve top complaint aspects for a specific product among negative reviews.
    Returns:
        List of dicts with aspect, display_name, count, and percentage.
    """
    row = _analytics_engine.find_product(product_id)
    if row is None:
        raise ValueError(f"Product '{product_id}' not found in product insights database.")

    aspects_json = row.get("top_complaint_aspects_json", "[]")
    if isinstance(aspects_json, str):
        return json.loads(aspects_json)
    return aspects_json if isinstance(aspects_json, list) else []


def get_product_summary(product_id: str) -> str:
    """
    Generate clean, human-readable customer feedback summary for a product.
    Matches the exact layout specified in user instructions:
    ---------------------------------------------------------
    Product: XYZ Headphones

    Total Reviews: 842

    Positive: 61%
    Neutral: 12%
    Negative: 27%

    Top complaints:
    1. Battery — 42%
    2. Sound Quality — 31%
    3. Durability — 18%
    4. Delivery — 9%
    ---------------------------------------------------------
    """
    row = _analytics_engine.find_product(product_id)
    if row is None:
        return f"Error: Product '{product_id}' was not found in the FeedbackIQ database."

    title = row["title"]
    # Truncate title cleanly if too long
    display_title = title if len(title) <= 80 else title[:77] + "..."

    total = int(row["total_reviews"])
    pos = int(round(row["positive_percentage"]))
    neu = int(round(row["neutral_percentage"]))
    neg = int(round(row["negative_percentage"]))

    aspects = get_product_aspects(product_id)

    lines = [
        f"Product: {display_title}",
        f"Domain: {row['domain']} | ASIN: {row['parent_asin']} | Rating: {row['average_rating']:.1f} / 5.0",
        "",
        f"Total Reviews: {total}",
        "",
        f"Positive: {pos}%",
        f"Neutral: {neu}%",
        f"Negative: {neg}%",
        "",
        "Top complaints:",
    ]

    if aspects:
        for idx, asp in enumerate(aspects[:5], 1):
            pct = int(round(asp["percentage"]))
            name = asp.get("display_name", asp["aspect"].replace("_", " ").title())
            lines.append(f"{idx}. {name} — {pct}%")
    else:
        lines.append("No significant complaint patterns detected.")

    return "\n".join(lines)


if __name__ == "__main__":
    df_insights = build_product_insights()
    print("\nProduct Insights Generated Successfully:")
    print(df_insights[["parent_asin", "domain", "total_reviews", "average_rating", "positive_percentage", "negative_percentage"]])

    # Test sample product query
    sample_pid = df_insights.iloc[0]["parent_asin"]
    print("\n" + "=" * 60)
    print("TESTING get_product_summary():")
    print("=" * 60)
    print(get_product_summary(sample_pid))
