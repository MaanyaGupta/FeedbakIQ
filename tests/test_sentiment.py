"""
Unit tests for Sentiment Analysis Baseline Pipeline (src/sentiment.py).
"""
import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

from src.sentiment import (
    prepare_text_features,
    split_dataset,
    build_baseline_pipeline,
    evaluate_sentiment_model,
    save_trained_model,
    train_and_evaluate_domain,
    CLASS_NAMES,
)


class TestSentimentBaseline(unittest.TestCase):
    def setUp(self):
        # Create a balanced miniature dataset
        np.random.seed(42)
        records = []
        for i in range(30):
            records.append({
                "domain": "Electronics",
                "title": "Great sound",
                "text": "Excellent headphones with clear audio and bass.",
                "sentiment": "positive",
                "rating": 5.0
            })
        for i in range(15):
            records.append({
                "domain": "Electronics",
                "title": "Broke soon",
                "text": "Terrible product, stopped working after two days.",
                "sentiment": "negative",
                "rating": 1.0
            })
        for i in range(15):
            records.append({
                "domain": "Electronics",
                "title": "Average",
                "text": "It is an okay item, neither great nor bad.",
                "sentiment": "neutral",
                "rating": 3.0
            })
        self.mock_df = pd.DataFrame(records)

    def test_prepare_text_features(self):
        X, y = prepare_text_features(self.mock_df)
        self.assertEqual(len(X), len(self.mock_df))
        self.assertEqual(len(y), len(self.mock_df))
        self.assertIn("Great sound. Excellent headphones", X.iloc[0])
        self.assertEqual(y.iloc[0], "positive")

    def test_split_dataset_stratification(self):
        X, y = prepare_text_features(self.mock_df)
        splits = split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42)

        # Total 60: 70% Train (~42), 15% Val (~9), 15% Test (~9)
        self.assertEqual(len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]), 60)
        
        # Verify all classes exist in train, val, and test splits
        for split_key in ["y_train", "y_val", "y_test"]:
            unique_classes = set(splits[split_key].unique())
            self.assertEqual(unique_classes, set(CLASS_NAMES))

    def test_train_and_evaluate_pipeline(self):
        X, y = prepare_text_features(self.mock_df)
        splits = split_dataset(X, y, test_size=0.20, val_size=0.20, random_state=42)

        pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=500, random_state=42)
        pipeline.fit(splits["X_train"], splits["y_train"])

        eval_results = evaluate_sentiment_model(pipeline, splits["X_test"], splits["y_test"])
        
        self.assertIn("accuracy", eval_results)
        self.assertIn("weighted_f1", eval_results)
        self.assertIn("macro_f1", eval_results)
        self.assertIn("confusion_matrix", eval_results)
        self.assertEqual(len(eval_results["confusion_matrix"]), 3)
        self.assertIn("neutral_f1", eval_results)
        self.assertGreaterEqual(eval_results["accuracy"], 0.0)
        self.assertLessEqual(eval_results["accuracy"], 1.0)

    def test_save_and_load_model(self):
        X, y = prepare_text_features(self.mock_df)
        pipeline = build_baseline_pipeline(ngram_range=(1, 1), max_features=100, random_state=42)
        pipeline.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = save_trained_model(pipeline, domain="test_domain", output_dir=Path(tmp_dir))
            self.assertTrue(save_path.exists())
            
            import joblib
            loaded = joblib.load(save_path)
            sample_pred = loaded.predict(["Excellent product, loved it!"])
            self.assertEqual(len(sample_pred), 1)
            self.assertIn(sample_pred[0], CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
