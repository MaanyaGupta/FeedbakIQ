"""
Unit tests for Product-Level Customer Feedback Analytics Layer (src/product_analytics.py).
Tests sentiment metrics calculation, aspect querying, and formatted product summary generation.
"""

import unittest
import pandas as pd
import tempfile
from pathlib import Path

from src.product_analytics import (
    get_product_sentiment,
    get_product_aspects,
    get_product_summary,
    ProductAnalytics,
    _analytics_engine,
)


class TestProductAnalytics(unittest.TestCase):

    def test_analytics_engine_loaded(self):
        df = _analytics_engine.data
        self.assertFalse(df.empty)
        self.assertIn("parent_asin", df.columns)
        self.assertIn("total_reviews", df.columns)
        self.assertIn("positive_percentage", df.columns)
        self.assertIn("neutral_percentage", df.columns)
        self.assertIn("negative_percentage", df.columns)
        self.assertIn("average_rating", df.columns)

    def test_get_product_sentiment(self):
        sample_id = _analytics_engine.data.iloc[0]["parent_asin"]
        sentiment_data = get_product_sentiment(sample_id)

        self.assertEqual(sentiment_data["parent_asin"], sample_id)
        self.assertIn("total_reviews", sentiment_data)
        self.assertIn("positive_reviews", sentiment_data)
        self.assertIn("neutral_reviews", sentiment_data)
        self.assertIn("negative_reviews", sentiment_data)
        self.assertGreaterEqual(sentiment_data["positive_percentage"], 0.0)
        self.assertLessEqual(sentiment_data["positive_percentage"], 100.0)
        self.assertGreater(sentiment_data["average_rating"], 0.0)

    def test_get_product_aspects(self):
        sample_id = _analytics_engine.data.iloc[0]["parent_asin"]
        aspects = get_product_aspects(sample_id)
        self.assertIsInstance(aspects, list)
        if aspects:
            first = aspects[0]
            self.assertIn("aspect", first)
            self.assertIn("count", first)
            self.assertIn("percentage", first)

    def test_get_product_summary_format(self):
        sample_id = _analytics_engine.data.iloc[0]["parent_asin"]
        summary = get_product_summary(sample_id)
        self.assertIsInstance(summary, str)
        self.assertIn("Product:", summary)
        self.assertIn("Total Reviews:", summary)
        self.assertIn("Positive:", summary)
        self.assertIn("Negative:", summary)

    def test_invalid_product_id(self):
        with self.assertRaises(ValueError):
            get_product_sentiment("NON_EXISTENT_ASIN_99999")


if __name__ == "__main__":
    unittest.main()
