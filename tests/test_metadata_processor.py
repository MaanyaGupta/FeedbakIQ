"""
Unit tests for product metadata processing, cleaning, joining, and filtering.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.config import MIN_REVIEWS_PER_PRODUCT
from src.metadata_processor import (
    clean_metadata,
    validate_and_join_reviews_with_metadata,
    analyze_and_filter_products,
    verify_parquet_file,
)


class TestMetadataProcessor(unittest.TestCase):
    def setUp(self):
        # Sample review dataframe
        self.sample_reviews = pd.DataFrame([
            {"domain": "Electronics", "parent_asin": "P1", "asin": "A1", "rating": 5.0, "sentiment": "positive", "title": "Great", "text": "Loved it!", "timestamp": 1000, "helpful_vote": 2, "verified_purchase": True},
            {"domain": "Electronics", "parent_asin": "P1", "asin": "A1", "rating": 4.0, "sentiment": "positive", "title": "Good", "text": "Pretty good", "timestamp": 1001, "helpful_vote": 0, "verified_purchase": True},
            {"domain": "Electronics", "parent_asin": "P2", "asin": "A2", "rating": 1.0, "sentiment": "negative", "title": "Bad", "text": "Broke soon", "timestamp": 1002, "helpful_vote": 1, "verified_purchase": False},
            {"domain": "Electronics", "parent_asin": "P3", "asin": "A3", "rating": 3.0, "sentiment": "neutral", "title": "OK", "text": "Average item", "timestamp": 1003, "helpful_vote": 0, "verified_purchase": True},
        ])

        # Sample raw metadata dataframe with duplicates and missing records
        self.sample_metadata = pd.DataFrame([
            {"domain": "Electronics", "parent_asin": "P1", "title": "<b>Headphones</b>", "average_rating": 4.5, "price": "$29.99", "description": ["High quality", "Wireless"], "categories": ["Electronics", "Audio"], "store": "Brand A"},
            {"domain": "Electronics", "parent_asin": "P1", "title": "Duplicate P1", "average_rating": 4.5, "price": "29.99", "description": ["Dup"], "categories": ["Audio"], "store": "Brand A"}, # Duplicate
            {"domain": "Electronics", "parent_asin": "P2", "title": "Speaker", "average_rating": 2.0, "price": "None", "description": [], "categories": ["Electronics"], "store": "Brand B"},
            {"domain": "Electronics", "parent_asin": None, "title": "No ASIN", "average_rating": 3.0, "price": "10.00", "description": None, "categories": None, "store": ""}, # Missing parent_asin
        ])

    def test_clean_metadata(self):
        cleaned_meta, stats = clean_metadata(self.sample_metadata, domain="Electronics")
        
        # Verify deduplication & missing removal
        self.assertEqual(stats["raw_metadata_count"], 4)
        self.assertEqual(stats["missing_parent_asin"], 1)
        self.assertEqual(stats["duplicate_parent_asin"], 1)
        self.assertEqual(stats["clean_metadata_count"], 2)
        self.assertEqual(len(cleaned_meta), 2)
        
        # Verify HTML cleaning & field normalization
        p1_row = cleaned_meta[cleaned_meta["parent_asin"] == "P1"].iloc[0]
        self.assertEqual(p1_row["title"], "Headphones")
        self.assertEqual(p1_row["description"], "High quality Wireless")
        self.assertEqual(p1_row["categories"], ["Electronics", "Audio"])
        self.assertEqual(p1_row["average_rating"], 4.5)

    def test_validate_and_join_no_row_multiplication(self):
        cleaned_meta, _ = clean_metadata(self.sample_metadata, domain="Electronics")
        df_joined, val_stats = validate_and_join_reviews_with_metadata(
            df_reviews=self.sample_reviews,
            df_meta=cleaned_meta,
            domain="Electronics"
        )
        
        # Verify join count is <= review count (no row multiplication)
        self.assertLessEqual(len(df_joined), len(self.sample_reviews))
        self.assertEqual(val_stats["duplicate_metadata_parent_asin"], 0)
        self.assertEqual(val_stats["unmatched_reviews_count"], 1) # P3 was unmatched
        self.assertEqual(len(df_joined), 3) # P1 (2 reviews) + P2 (1 review) = 3

    def test_analyze_and_filter_products(self):
        # Create dataset with one product having 25 reviews and another having 5 reviews
        p_many = ["P_MANY"] * 25
        p_few = ["P_FEW"] * 5
        asins = p_many + p_few
        
        df_revs = pd.DataFrame({
            "domain": "Electronics",
            "parent_asin": asins,
            "rating": 5.0,
            "sentiment": "positive",
            "title": "T",
            "text": "Text review sample long enough",
        })
        
        df_meta = pd.DataFrame([
            {"domain": "Electronics", "parent_asin": "P_MANY", "title": "Product Many", "average_rating": 4.8, "price": "50.00", "description": "Desc", "categories": ["Cat"], "store": "Store"},
            {"domain": "Electronics", "parent_asin": "P_FEW", "title": "Product Few", "average_rating": 3.0, "price": "10.00", "description": "Desc", "categories": ["Cat"], "store": "Store"},
        ])
        
        df_filt_rev, df_filt_prod, stats = analyze_and_filter_products(
            df_joined=df_revs,
            df_meta=df_meta,
            domain="Electronics",
            min_reviews=20
        )
        
        self.assertEqual(stats["reviews_before_product_filtering"], 30)
        self.assertEqual(stats["unique_products"], 2)
        self.assertEqual(stats["products_with_min_reviews"], 1)
        self.assertEqual(stats["reviews_remaining"], 25)
        self.assertEqual(stats["avg_reviews_per_product"], 25.0)
        self.assertEqual(stats["median_reviews_per_product"], 25.0)
        self.assertEqual(len(df_filt_prod), 1)
        self.assertEqual(df_filt_prod.iloc[0]["parent_asin"], "P_MANY")

    def test_verify_parquet_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_output.parquet"
            sample_df = pd.DataFrame({
                "col1": [1, 2],
                "col2": ["a", "b"]
            })
            sample_df.to_parquet(tmp_path, index=False)
            
            # Should pass without error
            res = verify_parquet_file(tmp_path, required_columns=["col1", "col2"])
            self.assertTrue(res)
            
            # Should raise ValueError if column missing
            with self.assertRaises(ValueError):
                verify_parquet_file(tmp_path, required_columns=["col1", "missing_col"])


if __name__ == "__main__":
    unittest.main()
