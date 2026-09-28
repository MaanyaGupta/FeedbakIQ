"""
Script to create the structured aspect annotation dataset (data/aspect_annotations.csv).
Samples 300 negative reviews (150 Electronics, 150 Fashion) from processed parquet datasets.
Bootstraps high-fidelity initial multi-label aspect annotations across domain taxonomies.
"""

import re
import logging
from pathlib import Path
from typing import List, Set, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_FILE = DATA_DIR / "aspect_annotations.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Taxonomies
COMMON_ASPECTS = ["packaging", "quality", "price", "delivery", "customer_service"]
ELECTRONICS_ASPECTS = COMMON_ASPECTS + ["battery", "performance", "compatibility", "durability"]
FASHION_ASPECTS = COMMON_ASPECTS + ["size_fit", "material", "design", "color", "durability"]

# Lexical matchers for initial annotation bootstrapping
ASPECT_PATTERNS = {
    # Electronics specific
    "battery": re.compile(r"\b(battery|batteries|charge|charging|charger|drain|dies|died|holds charge|power bank)\b", re.I),
    "performance": re.compile(r"\b(sound|audio|volume|bass|lag|laggy|slow|freeze|freezing|crash|glitch|speed|distort|muffled|static|buzz|screen|display|flicker|resolution|volume|hear|stream|buffering|touchscreen|remote)\b", re.I),
    "compatibility": re.compile(r"\b(compatible|compatibility|connect|connecting|connection|bluetooth|wifi|wi-fi|sync|pair|pairing|adapter|port|usb|hdmi|ios|android|mac|pc|cable)\b", re.I),
    "durability": re.compile(r"\b(durab|break|broke|broken|crack|cracked|stitching|shatter|fell apart|falling apart|tore|tear|snapped|peel|peeling|cheaply made|flimsy|fragile)\b", re.I),

    # Fashion specific
    "size_fit": re.compile(r"\b(size|sizing|fit|fits|fitting|tight|loose|small|smaller|large|larger|short|shorter|long|longer|waist|sleeves|shoulders|chest|snug|baggy|runs small|runs large|huge|tiny)\b", re.I),
    "material": re.compile(r"\b(material|fabric|cotton|polyester|synthetic|wool|silk|linen|thin|see through|see-through|rough|scratchy|itchy|softness|wrinkle|wrinkles|breathable)\b", re.I),
    "design": re.compile(r"\b(design|style|cut|shape|pattern|pockets|zipper|buttons|collar|look|looks|ugly|flattering|unflattering)\b", re.I),
    "color": re.compile(r"\b(color|colors|colour|shade|faded|fading|bleeding|dye|picture|photo|yellow|blue|red|black|white|darker|lighter|not as pictured|different color)\b", re.I),

    # Common
    "packaging": re.compile(r"\b(packag|box|package|boxed|bubble wrap|envelope|unopened|seal|dented|damaged box|open box)\b", re.I),
    "quality": re.compile(r"\b(quality|junk|garbage|trash|waste|poor|terrible|horrible|defective|cheap|worthless|shoddy|crap)\b", re.I),
    "price": re.compile(r"\b(price|pricing|cost|dollar|dollars|expensive|overpriced|value|money|worth it|waste of money|rip off|ripoff)\b", re.I),
    "delivery": re.compile(r"\b(deliver|delivery|ship|shipping|shipped|carrier|mail|arrived|late|transit|tracking|lost)\b", re.I),
    "customer_service": re.compile(r"\b(customer service|support|seller|return|returns|returned|refund|refunded|warranty|replacement|rep)\b", re.I),
}


def bootstrap_aspects(text: str, domain: str) -> List[str]:
    """Detect applicable aspects based on domain taxonomy and text patterns."""
    valid_aspects = ELECTRONICS_ASPECTS if domain.lower() == "electronics" else FASHION_ASPECTS
    matched: Set[str] = set()

    for aspect in valid_aspects:
        pattern = ASPECT_PATTERNS.get(aspect)
        if pattern and pattern.search(text):
            matched.add(aspect)

    # If no specific aspect matched, default to 'quality' for negative reviews
    if not matched:
        matched.add("quality")

    # If general negative words match without specific parts, ensure quality is recognized
    if any(w in text.lower() for w in ["junk", "trash", "garbage", "poor", "waste", "cheaply"]) and "quality" in valid_aspects:
        matched.add("quality")

    # Order aspects consistently
    return [a for a in valid_aspects if a in matched]


def create_annotation_dataset(
    target_count_per_domain: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample reviews and format structured annotation dataset."""
    elec_path = PROCESSED_DIR / "electronics_processed.parquet"
    fash_path = PROCESSED_DIR / "fashion_processed.parquet"

    logger.info("Reading processed parquet files...")
    df_elec = pd.read_parquet(elec_path, engine="fastparquet")
    df_fash = pd.read_parquet(fash_path, engine="fastparquet")

    # Filter negative reviews with informative length (35 - 750 chars)
    neg_elec = df_elec[
        (df_elec["sentiment"] == "negative")
        & (df_elec["text"].fillna("").str.len() >= 35)
        & (df_elec["text"].fillna("").str.len() <= 750)
    ].copy()

    neg_fash = df_fash[
        (df_fash["sentiment"] == "negative")
        & (df_fash["text"].fillna("").str.len() >= 35)
        & (df_fash["text"].fillna("").str.len() <= 750)
    ].copy()

    logger.info("Found %d negative Electronics and %d negative Fashion reviews.", len(neg_elec), len(neg_fash))

    sample_elec = neg_elec.sample(n=target_count_per_domain, random_state=random_state).reset_index(drop=True)
    sample_fash = neg_fash.sample(n=target_count_per_domain, random_state=random_state).reset_index(drop=True)

    rows = []

    # Process Electronics
    for i, row in sample_elec.iterrows():
        review_id = f"ELEC-{i + 1:03d}"
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        combined_text = f"{title}. {text}" if title and not text.startswith(title) else text
        aspects = bootstrap_aspects(combined_text, "Electronics")

        rows.append({
            "review_id": review_id,
            "domain": "Electronics",
            "text": combined_text,
            "aspect_labels": "; ".join(aspects),
        })

    # Process Fashion
    for i, row in sample_fash.iterrows():
        review_id = f"FASH-{i + 1:03d}"
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        combined_text = f"{title}. {text}" if title and not text.startswith(title) else text
        aspects = bootstrap_aspects(combined_text, "Fashion")

        rows.append({
            "review_id": review_id,
            "domain": "Fashion",
            "text": combined_text,
            "aspect_labels": "; ".join(aspects),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_FILE, index=False)
    logger.info("Successfully created aspect annotation file: %s (Total: %d reviews)", OUTPUT_FILE, len(df_out))
    return df_out


if __name__ == "__main__":
    df = create_annotation_dataset()
    print("\nAnnotation dataset summary:")
    print(df["domain"].value_counts())
    print("\nSample records:")
    print(df.head(4)[["review_id", "domain", "aspect_labels", "text"]])
