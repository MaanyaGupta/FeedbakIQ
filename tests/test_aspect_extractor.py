"""
Unit tests for Aspect Extraction Pipeline (src/aspect_extractor.py).
Tests taxonomy mappings, string parsing, metric computation, and frequency calculations.
"""

import unittest
import pandas as pd
from src.aspect_extractor import (
    COMMON_ASPECTS,
    ELECTRONICS_SPECIFIC,
    FASHION_SPECIFIC,
    DOMAIN_ASPECTS,
    parse_aspect_string,
    evaluate_aspect_extraction,
    compute_aspect_frequencies,
)


class TestAspectExtractor(unittest.TestCase):

    def test_taxonomies(self):
        # Common taxonomy
        self.assertIn("packaging", COMMON_ASPECTS)
        self.assertIn("quality", COMMON_ASPECTS)
        self.assertIn("price", COMMON_ASPECTS)
        self.assertIn("delivery", COMMON_ASPECTS)
        self.assertIn("customer_service", COMMON_ASPECTS)

        # Electronics taxonomy
        self.assertIn("battery", ELECTRONICS_SPECIFIC)
        self.assertIn("performance", ELECTRONICS_SPECIFIC)
        self.assertIn("compatibility", ELECTRONICS_SPECIFIC)
        self.assertIn("durability", ELECTRONICS_SPECIFIC)

        # Fashion taxonomy
        self.assertIn("size_fit", FASHION_SPECIFIC)
        self.assertIn("material", FASHION_SPECIFIC)
        self.assertIn("design", FASHION_SPECIFIC)
        self.assertIn("color", FASHION_SPECIFIC)
        self.assertIn("durability", FASHION_SPECIFIC)

        # Domain composite mappings
        elec_all = DOMAIN_ASPECTS["Electronics"]
        self.assertEqual(len(elec_all), 9)
        self.assertIn("battery", elec_all)
        self.assertIn("quality", elec_all)

        fash_all = DOMAIN_ASPECTS["Fashion"]
        self.assertEqual(len(fash_all), 10)
        self.assertIn("size_fit", fash_all)
        self.assertIn("material", fash_all)

    def test_parse_aspect_string(self):
        self.assertEqual(parse_aspect_string("battery; quality; performance"), ["battery", "quality", "performance"])
        self.assertEqual(parse_aspect_string("size_fit, material"), ["size_fit", "material"])
        self.assertEqual(parse_aspect_string(["battery", "durability"]), ["battery", "durability"])
        self.assertEqual(parse_aspect_string(""), [])
        self.assertEqual(parse_aspect_string(None), [])

    def test_evaluate_aspect_extraction(self):
        y_true = [
            ["battery", "quality"],
            ["size_fit"],
            ["durability", "material"],
        ]
        y_pred = [
            ["battery", "performance"],  # TP battery, FP performance, FN quality
            ["size_fit"],                # TP size_fit
            ["durability", "material"],  # TP durability, TP material
        ]

        metrics = evaluate_aspect_extraction(y_true, y_pred)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertIn("micro_f1", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_aspect", metrics)

        # size_fit perfect
        self.assertEqual(metrics["per_aspect"]["size_fit"]["f1"], 1.0)
        # battery perfect recall, precision 1.0
        self.assertEqual(metrics["per_aspect"]["battery"]["precision"], 1.0)

    def test_compute_aspect_frequencies(self):
        df = pd.DataFrame([
            {"domain": "Electronics", "aspect_labels": "battery; quality"},
            {"domain": "Electronics", "aspect_labels": "battery; performance"},
            {"domain": "Fashion", "aspect_labels": "size_fit; material"},
            {"domain": "Fashion", "aspect_labels": "size_fit; quality"},
        ])

        freq_df = compute_aspect_frequencies(df, aspects_column="aspect_labels", domain_column="domain")
        self.assertFalse(freq_df.empty)

        # In Electronics, battery appears in 2 out of 2 reviews = 100%
        elec_battery = freq_df[(freq_df["domain"] == "Electronics") & (freq_df["aspect"] == "battery")]
        self.assertEqual(elec_battery.iloc[0]["count"], 2)
        self.assertEqual(elec_battery.iloc[0]["frequency_pct"], 100.0)

        # In Fashion, size_fit appears in 2 out of 2 reviews = 100%
        fash_size = freq_df[(freq_df["domain"] == "Fashion") & (freq_df["aspect"] == "size_fit")]
        self.assertEqual(fash_size.iloc[0]["count"], 2)
        self.assertEqual(fash_size.iloc[0]["frequency_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
