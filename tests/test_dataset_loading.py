"""
Unit tests for configuration, data loading pipeline, and preprocessing pipeline.
"""
import unittest
import pandas as pd
from src.config import (
    CATEGORY_CONFIG,
    METADATA_CONFIG,
    SAMPLE_SIZE,
    MIN_REVIEWS_PER_PRODUCT,
    DATASET_NAME,
)
from src.data_loader import load_reviews, sample_reviews, save_raw_data
from src.preprocessing import (
    strip_html_artifacts,
    normalize_whitespace,
    cap_review_length,
    clean_text,
    assign_sentiment,
    preprocess_reviews,
)


class TestConfiguration(unittest.TestCase):
    def test_category_config_keys(self):
        self.assertIn("Electronics", CATEGORY_CONFIG)
        self.assertIn("Fashion", CATEGORY_CONFIG)

    def test_verified_config_names(self):
        # Exact verified HF dataset config names
        self.assertEqual(CATEGORY_CONFIG["Electronics"], "raw_review_Electronics")
        self.assertEqual(CATEGORY_CONFIG["Fashion"], "raw_review_Clothing_Shoes_and_Jewelry")

    def test_sample_and_filter_settings(self):
        self.assertEqual(SAMPLE_SIZE, 50000)
        self.assertEqual(MIN_REVIEWS_PER_PRODUCT, 20)


class TestPreprocessing(unittest.TestCase):
    def test_strip_html_and_entities(self):
        html_input = "Great product! <br><br>Fast delivery &amp; good price."
        expected = "Great product! Fast delivery & good price."
        cleaned = strip_html_artifacts(html_input)
        normalized = normalize_whitespace(cleaned)
        self.assertEqual(normalized, expected)

    def test_whitespace_normalization(self):
        raw = "  This   is   a   test \n\n with tabs\tand spaces.  "
        expected = "This is a test with tabs and spaces."
        self.assertEqual(normalize_whitespace(raw), expected)

    def test_cap_review_length(self):
        long_text = "a" * 1500
        capped = cap_review_length(long_text, max_chars=1000)
        self.assertEqual(len(capped), 1000)

    def test_preserve_punctuation_and_stopwords(self):
        sample = "Don't buy this! It does NOT work, and I'm very disappointed..."
        cleaned = clean_text(sample)
        # Verify negations and punctuation are intact
        self.assertIn("Don't", cleaned)
        self.assertIn("NOT", cleaned)
        self.assertIn("!", cleaned)
        self.assertIn("...", cleaned)

    def test_sentiment_mapping(self):
        self.assertEqual(assign_sentiment(1.0), "negative")
        self.assertEqual(assign_sentiment(2.0), "negative")
        self.assertEqual(assign_sentiment(3.0), "neutral")
        self.assertEqual(assign_sentiment(4.0), "positive")
        self.assertEqual(assign_sentiment(5.0), "positive")
        self.assertIsNone(assign_sentiment(None))

    def test_preprocess_reviews_end_to_end(self):
        sample_df = pd.DataFrame([
            {"text": "Excellent headphones! Clear audio and comfortable.", "rating": 5.0, "parent_asin": "P1", "domain": "Electronics"},
            {"text": "Excellent headphones! Clear audio and comfortable.", "rating": 5.0, "parent_asin": "P1", "domain": "Electronics"}, # Duplicate
            {"text": "", "rating": 1.0, "parent_asin": "P2", "domain": "Electronics"}, # Missing
            {"text": "ok", "rating": 3.0, "parent_asin": "P3", "domain": "Electronics"}, # Short (<3 words)
            {"text": "Broke immediately. Poor quality.", "rating": 1.0, "parent_asin": "P4", "domain": "Electronics"}, # Negative
            {"text": "Average device. Works fine enough.", "rating": 3.0, "parent_asin": "P5", "domain": "Electronics"}, # Neutral
        ])
        cleaned_df, stats = preprocess_reviews(sample_df, domain="Electronics", min_words=3, min_chars=10)
        self.assertEqual(stats["reviews_before"], 6)
        self.assertEqual(stats["missing_removed"], 1)
        self.assertEqual(stats["short_removed"], 1)
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["reviews_after"], 3)
        self.assertEqual(len(cleaned_df), 3)
        self.assertIn("sentiment", cleaned_df.columns)
        self.assertIn("domain", cleaned_df.columns)
        self.assertIn("parent_asin", cleaned_df.columns)


class TestLiveDatasetLoading(unittest.TestCase):
    def test_load_and_sample_electronics(self):
        """Test streaming 2 reviews from Electronics configuration."""
        ds = load_reviews("Electronics", streaming=True)
        df = sample_reviews(ds, domain="Electronics", sample_size=2)
        self.assertEqual(len(df), 2)
        self.assertIn("rating", df.columns)
        self.assertIn("text", df.columns)
        self.assertIn("domain", df.columns)
        self.assertEqual(df.iloc[0]["domain"], "Electronics")

    def test_load_and_sample_fashion(self):
        """Test streaming 2 reviews from Fashion configuration."""
        ds = load_reviews("Fashion", streaming=True)
        df = sample_reviews(ds, domain="Fashion", sample_size=2)
        self.assertEqual(len(df), 2)
        self.assertIn("rating", df.columns)
        self.assertIn("text", df.columns)
        self.assertIn("domain", df.columns)
        self.assertEqual(df.iloc[0]["domain"], "Fashion")


if __name__ == "__main__":
    unittest.main()
