import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

import plot_research_results as plotting  # noqa: E402


class ResearchPlottingTests(unittest.TestCase):
    def test_validated_summaries_preserve_phase7_protocol_values(self):
        paired = plotting.load_phase7_paired_results()
        leakage = plotting.summarize_leakage(paired)
        id3_mi_macro = leakage[
            (leakage["classifier"] == "ID3")
            & (leakage["selector"] == "MI")
            & (leakage["metric"] == "Macro-F1")
        ].iloc[0]
        self.assertAlmostEqual(
            id3_mi_macro["mean_difference"],
            0.02876913032579633,
        )
        self.assertEqual(id3_mi_macro["n_folds"], 15)

        model = plotting.load_phase7_model_results()
        model_summary = plotting.summarize_model_comparison(model)
        anova_macro = model_summary[
            (model_summary["selector"] == "ANOVA")
            & (model_summary["metric"] == "Macro-F1")
        ].iloc[0]
        self.assertAlmostEqual(
            anova_macro["mean_difference"],
            0.08299984797645889,
        )

    def test_figures_are_generated_without_the_restricted_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = plotting.plot_all(Path(directory))
            self.assertEqual(
                [path.name for path in paths],
                [
                    "leakage_comparison.png",
                    "model_comparison.png",
                    "feature_stability.png",
                ],
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_missing_values_are_rejected(self):
        frame = pd.read_csv(ROOT / "results/tables/feature_stability.csv")
        frame.loc[0, "mean_jaccard"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature_stability.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing values"):
                plotting.load_feature_stability(path)

    def test_duplicate_phase7_fold_configuration_is_rejected(self):
        frame = pd.read_csv(ROOT / "results/tables/phase7_paired_differences.csv")
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase7_paired_differences.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate fold rows"):
                plotting.load_phase7_paired_results(path)

    def test_non_finite_stability_summary_is_rejected(self):
        frame = pd.read_csv(ROOT / "results/tables/feature_stability.csv")
        frame.loc[0, "std_jaccard"] = float("inf")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature_stability.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "finite numeric values"):
                plotting.load_feature_stability(path)

    def test_duplicate_feature_stability_selector_is_rejected(self):
        frame = pd.read_csv(ROOT / "results/tables/feature_stability.csv")
        frame.loc[1, "selector"] = frame.loc[0, "selector"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature_stability.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate selectors"):
                plotting.load_feature_stability(path)


if __name__ == "__main__":
    unittest.main()
