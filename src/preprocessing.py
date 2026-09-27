"""
Review Preprocessing Pipeline for Amazon Reviews 2023.
Handles cleaning, normalization, deduplication, length capping,
and sentiment labeling for Electronics and Fashion domains.
"""
import html
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

try:
    from src.config import (
        RAW_CATEGORY_DIRS,
        PROCESSED_DATA_DIR,
        PROCESSED_CATEGORY_FILES,
        MIN_REVIEW_WORDS,
        MIN_REVIEW_CHARS,
        MAX_REVIEW_LENGTH,
    )
except ImportError:
    from config import (
        RAW_CATEGORY_DIRS,
        PROCESSED_DATA_DIR,
        PROCESSED_CATEGORY_FILES,
        MIN_REVIEW_WORDS,
        MIN_REVIEW_CHARS,
        MAX_REVIEW_LENGTH,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def strip_html_artifacts(text: str) -> str:
    """
    Unescape HTML entities (&amp;, &quot;, &#39;, etc.)
    and remove HTML tags (<br />, <div>, etc.).
    """
    if not isinstance(text, str):
        return ""
    # 1. Unescape HTML entities
    unescaped = html.unescape(text)
    # 2. Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", unescaped)
    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters (spaces, tabs, newlines)
    with a single space and strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def cap_review_length(text: str, max_chars: int = MAX_REVIEW_LENGTH) -> str:
    """
    Cap review text length to max_chars characters while preserving punctuation.
    """
    if not isinstance(text, str):
        return ""
    if len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def clean_text(text: str, max_chars: int = MAX_REVIEW_LENGTH) -> str:
    """
    Full text cleaning routine:
    - Strips HTML tags and entities
    - Normalizes whitespace
    - Caps review length
    - Preserves useful punctuation (!, ?, ., ,, quotes, apostrophes)
    - Preserves stopwords (critical for sentiment negations like 'not', 'no', 'hardly')
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    text = strip_html_artifacts(text)
    text = normalize_whitespace(text)
    text = cap_review_length(text, max_chars=max_chars)
    return text


def assign_sentiment(rating: Any) -> Optional[str]:
    """
    Create sentiment labels from rating:
    1–2 stars → negative
    3 stars   → neutral
    4–5 stars → positive
    """
    try:
        val = float(rating)
    except (ValueError, TypeError):
        return None

    if val <= 2.0:
        return "negative"
    elif val == 3.0:
        return "neutral"
    elif val >= 4.0:
        return "positive"
    return None


def preprocess_reviews(
    df: pd.DataFrame,
    domain: str,
    min_words: int = MIN_REVIEW_WORDS,
    min_chars: int = MIN_REVIEW_CHARS,
    max_chars: int = MAX_REVIEW_LENGTH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end cleaning and labeling for a single domain DataFrame:
    1. Count initial reviews.
    2. Remove missing review text.
    3. Clean text (HTML strip, whitespace normalization, length cap, preserve punctuation & stopwords).
    4. Remove empty and extremely short reviews.
    5. Remove duplicate reviews.
    6. Assign sentiment labels while keeping original rating.
    7. Preserve domain, parent_asin, and other critical columns.
    """
    stats: Dict[str, Any] = {
        "domain": domain,
        "reviews_before": len(df),
        "missing_removed": 0,
        "short_removed": 0,
        "duplicates_removed": 0,
        "reviews_after": 0,
        "sentiment_counts": {},
        "sentiment_percentages": {},
    }

    df_clean = df.copy()

    # Ensure rating is numeric
    if "rating" in df_clean.columns:
        df_clean["rating"] = pd.to_numeric(df_clean["rating"], errors="coerce")

    # Ensure domain is set
    if "domain" not in df_clean.columns:
        df_clean["domain"] = domain

    # 1. Handle missing review text
    missing_mask = df_clean["text"].isna() | (df_clean["text"].astype(str).str.strip() == "")
    stats["missing_removed"] = int(missing_mask.sum())
    df_clean = df_clean[~missing_mask].copy()

    # 2. Clean text & title
    df_clean["text"] = df_clean["text"].apply(lambda t: clean_text(t, max_chars=max_chars))
    if "title" in df_clean.columns:
        df_clean["title"] = df_clean["title"].apply(lambda t: clean_text(t, max_chars=max_chars) if pd.notna(t) else "")

    # 3. Remove empty and extremely short reviews
    word_counts = df_clean["text"].str.split().str.len()
    char_counts = df_clean["text"].str.len()
    short_mask = (word_counts < min_words) | (char_counts < min_chars)
    stats["short_removed"] = int(short_mask.sum())
    df_clean = df_clean[~short_mask].copy()

    # 4. Remove duplicate reviews
    dup_mask = df_clean.duplicated(subset=["text"], keep="first")
    stats["duplicates_removed"] = int(dup_mask.sum())
    df_clean = df_clean[~dup_mask].copy()

    # 5. Create sentiment labels from the rating (keep original rating)
    df_clean["sentiment"] = df_clean["rating"].apply(assign_sentiment)
    # Remove any row where rating could not be resolved to sentiment
    df_clean = df_clean[df_clean["sentiment"].notna()].copy()

    # 6. Final statistics
    stats["reviews_after"] = len(df_clean)
    sent_counts = df_clean["sentiment"].value_counts().to_dict()
    stats["sentiment_counts"] = sent_counts
    
    total = stats["reviews_after"]
    stats["sentiment_percentages"] = {
        label: (count / total * 100 if total > 0 else 0.0)
        for label, count in sent_counts.items()
    }

    # Ensure critical columns preserved
    expected_preservations = ["rating", "sentiment", "domain", "parent_asin"]
    for col in expected_preservations:
        if col not in df_clean.columns:
            logger.warning("Expected column '%s' missing from cleaned DataFrame!", col)

    return df_clean.reset_index(drop=True), stats


def load_raw_dataset(domain: str) -> pd.DataFrame:
    """Load raw sampled reviews parquet file for a domain."""
    raw_dir = RAW_CATEGORY_DIRS.get(domain)
    if not raw_dir:
        raise ValueError(f"Unknown domain: {domain}")
    
    parquet_path = raw_dir / "raw_sample.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Raw sample file not found at '{parquet_path}'. Please run src/data_loader.py first."
        )
    return pd.read_parquet(parquet_path)


def save_processed_dataset(df: pd.DataFrame, domain: str) -> Path:
    """Save cleaned and preprocessed dataset to data/processed directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = PROCESSED_CATEGORY_FILES.get(
        domain,
        PROCESSED_DATA_DIR / f"{domain.lower()}_processed.parquet"
    )
    df.to_parquet(target_path, index=False)
    logger.info("Saved processed '%s' data (%d reviews) to %s", domain, len(df), target_path)
    return target_path


def run_preprocessing_pipeline() -> Dict[str, Dict[str, Any]]:
    """
    Execute preprocessing pipeline across Electronics and Fashion domains,
    save the processed datasets, and print comprehensive statistics.
    """
    domains = ["Electronics", "Fashion"]
    all_stats: Dict[str, Dict[str, Any]] = {}
    processed_dfs: List[pd.DataFrame] = []

    for domain in domains:
        logger.info("Starting preprocessing for domain '%s'...", domain)
        raw_df = load_raw_dataset(domain)
        cleaned_df, stats = preprocess_reviews(raw_df, domain=domain)
        save_processed_dataset(cleaned_df, domain=domain)
        all_stats[domain] = stats
        processed_dfs.append(cleaned_df)

    # Save combined processed dataset
    if processed_dfs:
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        combined_path = PROCESSED_CATEGORY_FILES["Combined"]
        combined_df.to_parquet(combined_path, index=False)
        logger.info("Saved combined processed dataset (%d reviews) to %s", len(combined_df), combined_path)

    # Display requested statistics
    print_preprocessing_report(all_stats)
    return all_stats


def print_preprocessing_report(stats_dict: Dict[str, Dict[str, Any]]) -> None:
    """Print the exact formatting of preprocessing and sentiment statistics."""
    for domain, stats in stats_dict.items():
        print("\n" + "=" * 50)
        print(f"{domain.upper()} PREPROCESSING STATISTICS")
        print("=" * 50)
        print(f"Reviews before cleaning:  {stats['reviews_before']:,}")
        print(f"Reviews after cleaning:   {stats['reviews_after']:,}")
        print(f"Duplicates removed:       {stats['duplicates_removed']:,}")
        print(f"Missing reviews removed:  {stats['missing_removed']:,}")
        print(f"Short reviews removed:    {stats['short_removed']:,}")

        print(f"\n{domain} Sentiment Distribution:")
        counts = stats["sentiment_counts"]
        pcts = stats["sentiment_percentages"]
        for sentiment in ["positive", "neutral", "negative"]:
            count = counts.get(sentiment, 0)
            pct = pcts.get(sentiment, 0.0)
            print(f"- {sentiment.capitalize():<8}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    run_preprocessing_pipeline()
