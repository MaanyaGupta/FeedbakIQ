"""
Unit tests for Transformer-based Sentiment Analysis Pipeline (Phase 3).
Tests dataset formatting, stratified split consistency, metric computation,
evaluation parsing, and label mappings.
"""
import unittest
import numpy as np
import pandas as pd
import torch

from src.transformer_sentiment import (
    LABEL2ID,
    ID2LABEL,
    CLASS_NAMES,
    ReviewsDataset,
    split_data,
    compute_metrics_fn,
    evaluate_predictions,
)


class TestTransformerSentiment(unittest.TestCase):

    def test_label_mappings(self):
        self.assertEqual(LABEL2ID["negative"], 0)
        self.assertEqual(LABEL2ID["neutral"], 1)
        self.assertEqual(LABEL2ID["positive"], 2)
        self.assertEqual(ID2LABEL[0], "negative")
        self.assertEqual(ID2LABEL[1], "neutral")
        self.assertEqual(ID2LABEL[2], "positive")
        self.assertEqual(CLASS_NAMES, ["negative", "neutral", "positive"])

    def test_reviews_dataset(self):
        encodings = {
            "input_ids": torch.tensor([[101, 2054, 102], [101, 1037, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        labels = [0, 2]
        dataset = ReviewsDataset(encodings=encodings, labels=labels)
        self.assertEqual(len(dataset), 2)
        item0 = dataset[0]
        self.assertIn("input_ids", item0)
        self.assertIn("attention_mask", item0)
        self.assertIn("labels", item0)
        self.assertEqual(item0["labels"].item(), 0)

    def test_split_data_proportions_and_stratification(self):
        # 100 samples with 70 positive, 20 negative, 10 neutral
        texts = pd.Series([f"Sample review text number {i}" for i in range(100)])
        labels = pd.Series([2] * 70 + [0] * 20 + [1] * 10)

        splits = split_data(texts, labels, test_size=0.15, val_size=0.15, random_state=42)
        self.assertEqual(len(splits["train_texts"]), 70)
        self.assertEqual(len(splits["val_texts"]), 15)
        self.assertEqual(len(splits["test_texts"]), 15)

        # Verify all classes present in train
        train_labels = set(splits["train_labels"])
        self.assertIn(0, train_labels)
        self.assertIn(1, train_labels)
        self.assertIn(2, train_labels)

    def test_compute_metrics_fn(self):
        logits = np.array([
            [2.5, 0.1, -1.0],  # argmax 0
            [-0.5, 3.0, 0.2],  # argmax 1
            [-1.0, -0.2, 2.8], # argmax 2
        ])
        labels = np.array([0, 1, 2])
        metrics = compute_metrics_fn((logits, labels))

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)

    def test_evaluate_predictions_and_neutral_metrics(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 2, 2, 2]

        results = evaluate_predictions(y_true, y_pred, split_name="Test")
        self.assertEqual(results["sample_count"], 6)
        self.assertIn("accuracy", results)
        self.assertIn("neutral_precision", results)
        self.assertIn("neutral_recall", results)
        self.assertIn("neutral_f1", results)
        self.assertEqual(results["neutral_support"], 2)
        self.assertEqual(len(results["confusion_matrix"]), 3)


if __name__ == "__main__":
    unittest.main()
