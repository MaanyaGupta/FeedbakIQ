"""
Product Metadata Processing Pipeline for Amazon Reviews 2023.
Loads metadata for Electronics and Fashion domains, validates schemas,
deduplicates and cleans product metadata, joins reviews with metadata without
multiplying rows, filters products by MIN_REVIEWS_PER_PRODUCT, computes product-level
statistics, and saves the final processed reviews and products Parquet files.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import html
import re
import numpy as np
import pandas as pd

try:
    from src.config import (
        DATASET_NAME,
        CATEGORY_CONFIG,
        METADATA_CONFIG,
        PROCESSED_DATA_DIR,
        PROCESSED_CATEGORY_FILES,
        MIN_REVIEWS_PER_PRODUCT,
        PROCESSED_REVIEWS_FILE,
        PROCESSED_PRODUCTS_FILE,
        REQUIRED_REVIEW_OUTPUT_COLUMNS,
        REQUIRED_PRODUCT_OUTPUT_COLUMNS,
        METADATA_COLUMNS,
        SAMPLE_SIZE,
    )
    from src.data_loader import load_metadata, sample_metadata
    from src.preprocessing import clean_text, strip_html_artifacts, normalize_whitespace
except ImportError:
    from config import (
        DATASET_NAME,
        CATEGORY_CONFIG,
        METADATA_CONFIG,
        PROCESSED_DATA_DIR,
        PROCESSED_CATEGORY_FILES,
        MIN_REVIEWS_PER_PRODUCT,
        PROCESSED_REVIEWS_FILE,
        PROCESSED_PRODUCTS_FILE,
        REQUIRED_REVIEW_OUTPUT_COLUMNS,
        REQUIRED_PRODUCT_OUTPUT_COLUMNS,
        METADATA_COLUMNS,
        SAMPLE_SIZE,
    )
    from data_loader import load_metadata, sample_metadata
    from preprocessing import clean_text, strip_html_artifacts, normalize_whitespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_metadata(df_meta: pd.DataFrame, domain: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and normalize product metadata:
    - Check missing parent_asin & remove them
    - Check duplicate parent_asin & deduplicate metadata
    - Clean title, description, categories, price, store, average_rating
    - Ensure domain column is set
    """
    stats: Dict[str, Any] = {
        "domain": domain,
        "raw_metadata_count": len(df_meta),
        "missing_parent_asin": 0,
        "duplicate_parent_asin": 0,
        "clean_metadata_count": 0,
    }

    df_clean = df_meta.copy()
    if "domain" not in df_clean.columns:
        df_clean["domain"] = domain

    # 1. Check missing parent_asin
    if "parent_asin" in df_clean.columns:
        missing_mask = df_clean["parent_asin"].isna() | (df_clean["parent_asin"].astype(str).str.strip() == "")
        stats["missing_parent_asin"] = int(missing_mask.sum())
        df_clean = df_clean[~missing_mask].copy()
        df_clean["parent_asin"] = df_clean["parent_asin"].astype(str).str.strip()
    else:
        logger.warning("Column 'parent_asin' missing from metadata DataFrame for %s", domain)
        stats["missing_parent_asin"] = len(df_clean)

    # 2. Check duplicate metadata parent_asin & deduplicate
    dup_mask = df_clean.duplicated(subset=["parent_asin"], keep="first")
    stats["duplicate_parent_asin"] = int(dup_mask.sum())
    df_clean = df_clean.drop_duplicates(subset=["parent_asin"], keep="first").copy()

    # 3. Clean specific metadata fields
    if "title" in df_clean.columns:
        df_clean["title"] = df_clean["title"].apply(lambda t: clean_text(t) if pd.notna(t) else "")

    if "description" in df_clean.columns:
        def process_desc(d):
            if isinstance(d, (list, np.ndarray)):
                joined = " ".join([str(x) for x in d if pd.notna(x)])
                return clean_text(joined)
            elif isinstance(d, str):
                return clean_text(d)
            return ""
        df_clean["description"] = df_clean["description"].apply(process_desc)

    if "categories" in df_clean.columns:
        def process_cats(c):
            if isinstance(c, (list, np.ndarray)):
                return [str(x).strip() for x in c if pd.notna(x)]
            elif isinstance(c, str):
                return [c.strip()]
            return []
        df_clean["categories"] = df_clean["categories"].apply(process_cats)

    if "price" in df_clean.columns:
        def process_price(p):
            if pd.isna(p) or p is None or str(p).strip().lower() in ["none", "nan", "null"]:
                return "None"
            return str(p).strip()
        df_clean["price"] = df_clean["price"].apply(process_price)

    if "store" in df_clean.columns:
        df_clean["store"] = df_clean["store"].apply(lambda s: normalize_whitespace(str(s)) if pd.notna(s) else "")

    if "average_rating" in df_clean.columns:
        df_clean["average_rating"] = pd.to_numeric(df_clean["average_rating"], errors="coerce")

    stats["clean_metadata_count"] = len(df_clean)
    logger.info("Cleaned metadata for domain '%s': %d products retained (%d duplicates removed, %d missing removed).",
                domain, len(df_clean), stats["duplicate_parent_asin"], stats["missing_parent_asin"])
    
    return df_clean.reset_index(drop=True), stats


def validate_and_join_reviews_with_metadata(
    df_reviews: pd.DataFrame,
    df_meta: pd.DataFrame,
    domain: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform pre-join validation and join reviews with metadata.
    Checks:
    - duplicate metadata parent_asin
    - missing parent_asin in reviews and metadata
    - unmatched reviews
    Ensures join does NOT multiply review rows.
    """
    validation_stats: Dict[str, Any] = {
        "domain": domain,
        "reviews_count_before_join": len(df_reviews),
        "metadata_count_before_join": len(df_meta),
        "missing_parent_asin_in_reviews": 0,
        "missing_parent_asin_in_metadata": 0,
        "duplicate_metadata_parent_asin": 0,
        "unmatched_reviews_count": 0,
        "unmatched_unique_products": 0,
        "joined_reviews_count": 0,
    }

    # 1. Missing parent_asin check in reviews
    missing_rev_pasin = df_reviews["parent_asin"].isna() | (df_reviews["parent_asin"].astype(str).str.strip() == "")
    validation_stats["missing_parent_asin_in_reviews"] = int(missing_rev_pasin.sum())
    df_reviews_clean = df_reviews[~missing_rev_pasin].copy()

    # 2. Missing parent_asin check in metadata
    missing_meta_pasin = df_meta["parent_asin"].isna() | (df_meta["parent_asin"].astype(str).str.strip() == "")
    validation_stats["missing_parent_asin_in_metadata"] = int(missing_meta_pasin.sum())
    df_meta_clean = df_meta[~missing_meta_pasin].copy()

    # 3. Duplicate metadata parent_asin check & deduplication
    dup_meta_pasin = df_meta_clean.duplicated(subset=["parent_asin"], keep="first")
    validation_stats["duplicate_metadata_parent_asin"] = int(dup_meta_pasin.sum())
    if validation_stats["duplicate_metadata_parent_asin"] > 0:
        logger.warning("Found %d duplicate parent_asin in metadata for domain '%s'. Deduplicating...",
                       validation_stats["duplicate_metadata_parent_asin"], domain)
        df_meta_clean = df_meta_clean.drop_duplicates(subset=["parent_asin"], keep="first").copy()

    # 4. Unmatched reviews check
    review_asins = set(df_reviews_clean["parent_asin"])
    metadata_asins = set(df_meta_clean["parent_asin"])
    unmatched_asins = review_asins - metadata_asins
    validation_stats["unmatched_unique_products"] = len(unmatched_asins)

    unmatched_rev_mask = df_reviews_clean["parent_asin"].isin(unmatched_asins)
    validation_stats["unmatched_reviews_count"] = int(unmatched_rev_mask.sum())

    logger.info("[%s Pre-Join Check] Missing rev parent_asin: %d | Missing meta parent_asin: %d | Dup meta parent_asin: %d | Unmatched reviews: %d across %d products",
                domain,
                validation_stats["missing_parent_asin_in_reviews"],
                validation_stats["missing_parent_asin_in_metadata"],
                validation_stats["duplicate_metadata_parent_asin"],
                validation_stats["unmatched_reviews_count"],
                validation_stats["unmatched_unique_products"])

    # 5. Join reviews with metadata using: reviews.parent_asin = metadata.parent_asin
    meta_columns_to_merge = ["parent_asin", "title", "average_rating", "price", "description", "categories", "store"]
    # Rename product title column to product_title during merge if review dataframe already has title
    meta_sub = df_meta_clean[[col for col in meta_columns_to_merge if col in df_meta_clean.columns]].copy()
    meta_sub = meta_sub.rename(columns={"title": "product_title"})

    df_joined = df_reviews_clean.merge(
        meta_sub,
        on="parent_asin",
        how="inner",
    )

    validation_stats["joined_reviews_count"] = len(df_joined)

    # CRITICAL CHECK: Ensure join does NOT multiply review rows
    assert len(df_joined) <= len(df_reviews_clean), (
        f"Fatal error: Join multiplied review rows for {domain}! "
        f"Reviews before join: {len(df_reviews_clean)}, after join: {len(df_joined)}"
    )
    logger.info("[%s Join Verified] Join did NOT multiply review rows. Initial reviews: %d -> Joined: %d",
                domain, len(df_reviews_clean), len(df_joined))

    return df_joined, validation_stats


def analyze_and_filter_products(
    df_joined: pd.DataFrame,
    df_meta: pd.DataFrame,
    domain: str,
    min_reviews: int = MIN_REVIEWS_PER_PRODUCT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Analyze the number of reviews per product and filter products/reviews based on min_reviews (default: 20).
    Calculates exact required statistics:
    - Reviews before product filtering
    - Unique products
    - Products with >=20 reviews
    - Reviews remaining
    - Average reviews/product
    - Median reviews/product
    """
    reviews_before = len(df_joined)
    unique_products_before = df_joined["parent_asin"].nunique()

    product_review_counts = df_joined["parent_asin"].value_counts()
    valid_products_asins = product_review_counts[product_review_counts >= min_reviews].index

    # Filter reviews to keep only those from products with >= min_reviews
    df_filtered_reviews = df_joined[df_joined["parent_asin"].isin(valid_products_asins)].copy()

    # Filter products metadata to keep only valid products with >= min_reviews
    df_filtered_products = df_meta[df_meta["parent_asin"].isin(valid_products_asins)].copy()

    # Calculate statistics on remaining reviews per product
    counts_remaining = df_filtered_reviews["parent_asin"].value_counts()
    products_with_min_reviews = len(valid_products_asins)
    reviews_remaining = len(df_filtered_reviews)

    avg_reviews = float(counts_remaining.mean()) if len(counts_remaining) > 0 else 0.0
    median_reviews = float(counts_remaining.median()) if len(counts_remaining) > 0 else 0.0

    stats: Dict[str, Any] = {
        "domain": domain,
        "reviews_before_product_filtering": reviews_before,
        "unique_products": unique_products_before,
        "products_with_min_reviews": products_with_min_reviews,
        "reviews_remaining": reviews_remaining,
        "avg_reviews_per_product": avg_reviews,
        "median_reviews_per_product": median_reviews,
        "min_reviews_threshold": min_reviews,
    }

    return df_filtered_reviews.reset_index(drop=True), df_filtered_products.reset_index(drop=True), stats


def print_product_analysis_report(all_stats: Dict[str, Dict[str, Any]]) -> None:
    """Print the exact formatted summary of product metadata analysis."""
    for domain, stats in all_stats.items():
        print("\n" + "=" * 50)
        print(f"{domain.upper()} PRODUCT METADATA ANALYSIS")
        print("=" * 50)
        print(f"Reviews before product filtering: {stats['reviews_before_product_filtering']:,}")
        print(f"Unique products:                  {stats['unique_products']:,}")
        print(f"Products with >=20 reviews:       {stats['products_with_min_reviews']:,}")
        print(f"Reviews remaining:                {stats['reviews_remaining']:,}")
        print(f"Average reviews/product:          {stats['avg_reviews_per_product']:.2f}")
        print(f"Median reviews/product:           {stats['median_reviews_per_product']:.2f}")


def run_product_metadata_pipeline(
    sample_size: int = SAMPLE_SIZE,
    min_reviews: int = MIN_REVIEWS_PER_PRODUCT,
    domains: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute end-to-end product metadata processing pipeline:
    1. Load processed reviews for Electronics and Fashion.
    2. Load and clean metadata for both domains.
    3. Validate pre-join conditions (duplicate parent_asin, missing parent_asin, unmatched reviews).
    4. Join reviews with metadata (verifying no review row multiplication).
    5. Analyze reviews per product & filter products with >= min_reviews.
    6. Report required metrics.
    7. Save final data/processed/reviews.parquet and data/processed/products.parquet.
    8. Verify parquet files.
    """
    domains = domains or ["Electronics", "Fashion"]
    all_report_stats: Dict[str, Dict[str, Any]] = {}
    final_reviews_list: List[pd.DataFrame] = []
    final_products_list: List[pd.DataFrame] = []

    for domain in domains:
        logger.info("--- Processing Product Metadata for domain: '%s' ---", domain)

        # 1. Load processed reviews
        processed_review_file = PROCESSED_CATEGORY_FILES.get(
            domain, PROCESSED_DATA_DIR / f"{domain.lower()}_processed.parquet"
        )
        if not processed_review_file.exists():
            raise FileNotFoundError(
                f"Processed review file for '{domain}' not found at {processed_review_file}. "
                "Please run preprocessing pipeline first."
            )
        df_reviews = pd.read_parquet(processed_review_file)
        logger.info("Loaded %d processed reviews for '%s'.", len(df_reviews), domain)

        # 2. Load and clean product metadata
        raw_meta_ds = load_metadata(domain=domain, streaming=True)
        # Sample metadata records matching the unique parent_asins in df_reviews
        target_asins = set(df_reviews["parent_asin"].dropna())
        df_meta_raw = sample_metadata(
            dataset=raw_meta_ds,
            domain=domain,
            sample_size=sample_size,
            target_asins=target_asins,
        )
        df_meta_clean, meta_stats = clean_metadata(df_meta_raw, domain=domain)

        # 3. Validate and join reviews with metadata
        df_joined, val_stats = validate_and_join_reviews_with_metadata(
            df_reviews=df_reviews,
            df_meta=df_meta_clean,
            domain=domain,
        )

        # 4. Filter products and reviews by MIN_REVIEWS_PER_PRODUCT (20)
        df_filt_rev, df_filt_prod, report_stats = analyze_and_filter_products(
            df_joined=df_joined,
            df_meta=df_meta_clean,
            domain=domain,
            min_reviews=min_reviews,
        )

        all_report_stats[domain] = report_stats
        final_reviews_list.append(df_filt_rev)
        final_products_list.append(df_filt_prod)

    # 5. Report statistics for Electronics and Fashion
    print_product_analysis_report(all_report_stats)

    # 6. Save combined outputs to data/processed/reviews.parquet and data/processed/products.parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_all_reviews = pd.concat(final_reviews_list, ignore_index=True) if final_reviews_list else pd.DataFrame()
    df_all_products = pd.concat(final_products_list, ignore_index=True) if final_products_list else pd.DataFrame()

    # Ensure required review columns exist
    for col in REQUIRED_REVIEW_OUTPUT_COLUMNS:
        if col not in df_all_reviews.columns:
            df_all_reviews[col] = None
    df_all_reviews = df_all_reviews[REQUIRED_REVIEW_OUTPUT_COLUMNS].copy()

    # Ensure required product columns exist
    for col in REQUIRED_PRODUCT_OUTPUT_COLUMNS:
        if col not in df_all_products.columns:
            df_all_products[col] = None
    df_all_products = df_all_products[REQUIRED_PRODUCT_OUTPUT_COLUMNS].copy()

    # Save to Parquet
    df_all_reviews.to_parquet(PROCESSED_REVIEWS_FILE, index=False)
    logger.info("Saved final reviews dataset (%d rows) to %s", len(df_all_reviews), PROCESSED_REVIEWS_FILE)

    df_all_products.to_parquet(PROCESSED_PRODUCTS_FILE, index=False)
    logger.info("Saved final products dataset (%d rows) to %s", len(df_all_products), PROCESSED_PRODUCTS_FILE)

    # 7. Verification step: Read both Parquet files back
    verify_parquet_file(PROCESSED_REVIEWS_FILE, REQUIRED_REVIEW_OUTPUT_COLUMNS)
    verify_parquet_file(PROCESSED_PRODUCTS_FILE, REQUIRED_PRODUCT_OUTPUT_COLUMNS)

    return all_report_stats


def verify_parquet_file(file_path: Path, required_columns: List[str]) -> bool:
    """
    Verify that a Parquet file exists, can be read successfully by pandas,
    and contains all required columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Verification failed: File not found at '{file_path}'")
    
    df = pd.read_parquet(file_path)
    logger.info("Successfully read '%s': %d rows, %d columns.", file_path.name, len(df), len(df.columns))
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Verification failed for '{file_path.name}': Missing required columns {missing_cols}")
    
    print(f"VERIFICATION SUCCESS: '{file_path.name}' read successfully ({len(df):,} rows, columns: {list(df.columns)})")
    return True


if __name__ == "__main__":
    run_product_metadata_pipeline()
