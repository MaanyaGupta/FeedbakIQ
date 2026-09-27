"""
Data loading and sampling pipeline for Amazon Reviews 2023.
Loads reviews for Electronics and Fashion (Clothing, Shoes & Jewelry) domains,
samples configurable amounts, retains key schema attributes, and persists raw data.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from datasets import load_dataset, IterableDataset

try:
    from src.config import (
        DATASET_NAME,
        CATEGORY_CONFIG,
        METADATA_CONFIG,
        RAW_CATEGORY_DIRS,
        SAMPLE_SIZE,
        REVIEW_COLUMNS,
        METADATA_COLUMNS,
    )
except ImportError:
    from config import (
        DATASET_NAME,
        CATEGORY_CONFIG,
        METADATA_CONFIG,
        RAW_CATEGORY_DIRS,
        SAMPLE_SIZE,
        REVIEW_COLUMNS,
        METADATA_COLUMNS,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_reviews(
    domain: str,
    streaming: bool = True,
    trust_remote_code: bool = True,
) -> IterableDataset:
    """
    Load Amazon Reviews 2023 for a specified domain using verified config names.
    Defaults to streaming=True to avoid loading multi-gigabyte categories into memory.
    """
    if domain not in CATEGORY_CONFIG:
        raise ValueError(
            f"Unsupported domain '{domain}'. Configured domains: {list(CATEGORY_CONFIG.keys())}"
        )
    config_name = CATEGORY_CONFIG[domain]
    logger.info("Loading reviews for domain '%s' using config '%s' (streaming=%s)...", domain, config_name, streaming)
    
    dataset = load_dataset(
        DATASET_NAME,
        config_name,
        split="full",
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )
    return dataset


def sample_reviews(
    dataset: IterableDataset,
    domain: str,
    sample_size: int = SAMPLE_SIZE,
    fields: Optional[List[str]] = None,
    log_interval: int = 10000,
) -> pd.DataFrame:
    """
    Sample a configurable number of reviews from the streaming dataset.
    Retains specified fields and adds the 'domain' column.
    """
    target_fields = fields or REVIEW_COLUMNS
    logger.info("Sampling %d reviews for domain '%s'...", sample_size, domain)
    
    records = []
    loaded_count = 0
    
    for count, item in enumerate(dataset, start=1):
        loaded_count = count
        # Extract desired schema fields
        row = {field: item.get(field) for field in target_fields}
        # Add domain column
        row["domain"] = domain
        records.append(row)
        
        if count % log_interval == 0:
            logger.info("Domain '%s': sampled %d / %d reviews...", domain, count, sample_size)
            
        if count >= sample_size:
            break
            
    df = pd.DataFrame(records)
    logger.info(
        "Completed sampling for '%s': %d reviews collected across %d columns.",
        domain,
        len(df),
        len(df.columns),
    )
    return df


def load_metadata(
    domain: str,
    streaming: bool = True,
    trust_remote_code: bool = True,
) -> IterableDataset:
    """
    Load Amazon Reviews 2023 product metadata for a specified domain using verified config names.
    Defaults to streaming=True.
    """
    if domain not in METADATA_CONFIG:
        raise ValueError(
            f"Unsupported domain '{domain}'. Configured metadata domains: {list(METADATA_CONFIG.keys())}"
        )
    config_name = METADATA_CONFIG[domain]
    logger.info("Loading metadata for domain '%s' using config '%s' (streaming=%s)...", domain, config_name, streaming)
    
    dataset = load_dataset(
        DATASET_NAME,
        config_name,
        split="full",
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )
    return dataset


def sample_metadata(
    dataset: IterableDataset,
    domain: str,
    sample_size: int = SAMPLE_SIZE,
    fields: Optional[List[str]] = None,
    target_asins: Optional[set] = None,
    max_scan: Optional[int] = None,
    log_interval: int = 10000,
) -> pd.DataFrame:
    """
    Sample product metadata from the streaming dataset.
    Retains specified fields and adds the 'domain' column.
    If target_asins is provided, filters items to those matching target_asins.
    Deduplicates on parent_asin during sampling.
    """
    target_fields = fields or METADATA_COLUMNS
    max_scan_limit = max_scan or sample_size
    logger.info("Sampling metadata for domain '%s' (sample_size=%d, max_scan=%d)...", domain, sample_size, max_scan_limit)
    
    records = []
    seen_asins = set()
    
    for count, item in enumerate(dataset, start=1):
        pasin = item.get("parent_asin")
        if pasin and (target_asins is None or pasin in target_asins) and pasin not in seen_asins:
            seen_asins.add(pasin)
            row = {field: item.get(field) for field in target_fields}
            row["domain"] = domain
            records.append(row)
            
        if count % log_interval == 0:
            logger.info("Domain '%s': scanned %d metadata records, collected %d unique products...", domain, count, len(records))
            
        if len(records) >= sample_size or count >= max_scan_limit or (target_asins is not None and len(seen_asins) >= len(target_asins)):
            break
            
    df = pd.DataFrame(records)
    logger.info("Completed metadata sampling for '%s': %d products collected across %d columns.", domain, len(df), len(df.columns))
    return df


def save_raw_data(
    df: pd.DataFrame,
    domain: str,
    filename: str = "raw_sample.parquet",
) -> Path:
    """
    Save sampled raw reviews DataFrame under data/raw/<domain>/ directory.
    Supports Parquet (recommended), CSV, or JSON Lines based on file extension.
    """
    target_dir = RAW_CATEGORY_DIRS.get(domain)
    if not target_dir:
        raise ValueError(f"No raw directory configured for domain '{domain}'")
        
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename
    
    if filename.endswith(".parquet"):
        df.to_parquet(file_path, index=False)
    elif filename.endswith(".csv"):
        df.to_csv(file_path, index=False)
    elif filename.endswith(".jsonl") or filename.endswith(".json"):
        df.to_json(file_path, orient="records", lines=True)
    else:
        df.to_parquet(file_path, index=False)
        
    logger.info("Saved %d raw reviews for domain '%s' to '%s'", len(df), domain, file_path)
    return file_path


def run_data_loading_pipeline(
    sample_size: int = SAMPLE_SIZE,
    domains: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Orchestrate the full data loading pipeline:
    - Load streaming reviews
    - Sample configurable number of reviews per domain
    - Add domain column
    - Save raw data to respective directories
    """
    domains = domains or ["Electronics", "Fashion"]
    results = {}
    
    logger.info("Starting Data Loading Pipeline (sample_size=%d)...", sample_size)
    
    for domain in domains:
        print(f"\nProcessing domain: {domain}")
        # 1. Load reviews (streaming)
        dataset = load_reviews(domain, streaming=True)
        
        # 2. Sample reviews
        df_sampled = sample_reviews(dataset, domain=domain, sample_size=sample_size)
        
        # 3. Save raw data
        saved_path = save_raw_data(df_sampled, domain=domain)
        
        results[domain] = {
            "reviews_loaded": len(df_sampled),
            "reviews_sampled": len(df_sampled),
            "columns": list(df_sampled.columns),
            "saved_to": str(saved_path),
        }
        
    return results


if __name__ == "__main__":
    results = run_data_loading_pipeline(sample_size=SAMPLE_SIZE)
    
    print("\n" + "=" * 50)
    print("DATA LOADING PIPELINE SUMMARY")
    print("=" * 50)
    for domain, stats in results.items():
        print(f"\n{domain}:")
        print(f"- reviews loaded: {stats['reviews_loaded']:,}")
        print(f"- reviews sampled: {stats['reviews_sampled']:,}")
        print(f"- columns: {stats['columns']}")
        print(f"- saved to: {stats['saved_to']}")
