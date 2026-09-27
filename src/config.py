"""
Configuration settings for the Customer Feedback & Sentiment Analysis System.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Hugging Face Dataset Details
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

# Verified Hugging Face dataset configuration names:
# - Electronics: "raw_review_Electronics"
# - Fashion: Amazon category corresponds to "Clothing, Shoes & Jewelry", so config is "raw_review_Clothing_Shoes_and_Jewelry"
CATEGORY_CONFIG = {
    "Electronics": "raw_review_Electronics",
    "Fashion": "raw_review_Clothing_Shoes_and_Jewelry"
}

# Corresponding verified metadata configuration names
METADATA_CONFIG = {
    "Electronics": "raw_meta_Electronics",
    "Fashion": "raw_meta_Clothing_Shoes_and_Jewelry"
}

# Raw data storage subdirectories
RAW_CATEGORY_DIRS = {
    "Electronics": RAW_DATA_DIR / "electronics",
    "Fashion": RAW_DATA_DIR / "fashion"
}

# Processed data paths
PROCESSED_CATEGORY_FILES = {
    "Electronics": PROCESSED_DATA_DIR / "electronics_processed.parquet",
    "Fashion": PROCESSED_DATA_DIR / "fashion_processed.parquet",
    "Combined": PROCESSED_DATA_DIR / "combined_processed.parquet"
}

# Configurable Sampling and Filtering Parameters
SAMPLE_SIZE = 50000
MIN_REVIEWS_PER_PRODUCT = 20
RANDOM_STATE = 42

# Preprocessing & Text Cleaning Configurations
MIN_REVIEW_WORDS = 3
MIN_REVIEW_CHARS = 10
MAX_REVIEW_LENGTH = 1000  # Cap characters per review

# Expected Review Schema Fields
REVIEW_COLUMNS = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase"
]

# Expected Metadata Schema Fields
METADATA_COLUMNS = [
    "parent_asin",
    "title",
    "average_rating",
    "price",
    "description",
    "categories",
    "store"
]

# Required Processed Parquet Output File Paths
PROCESSED_REVIEWS_FILE = PROCESSED_DATA_DIR / "reviews.parquet"
PROCESSED_PRODUCTS_FILE = PROCESSED_DATA_DIR / "products.parquet"

# Required Schema for Final Output Files
REQUIRED_REVIEW_OUTPUT_COLUMNS = [
    "domain",
    "parent_asin",
    "asin",
    "rating",
    "sentiment",
    "title",
    "text",
    "timestamp",
    "helpful_vote",
    "verified_purchase"
]

REQUIRED_PRODUCT_OUTPUT_COLUMNS = [
    "domain",
    "parent_asin",
    "title",
    "average_rating",
    "price",
    "description",
    "categories",
    "store"
]

