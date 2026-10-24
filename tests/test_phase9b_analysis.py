import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze_phase7_results as analysis  # noqa: E402


class Phase9BAnalysisTests(unittest.TestCase):
    def test_bootstrap_interval_is_deterministic(self):
        differences = np.array([0.0, 0.0, 0.1, 0.2])
        first = analysis._bootstrap_interval(
            differences,
            np.random.default_rng(analysis.BOOTSTRAP_SEED),
            500,
        )
        second = analysis._bootstrap_interval(
            differences,
            np.random.default_rng(analysis.BOOTSTRAP_SEED),
            500,
        )
        self.assertEqual(first, second)

    def test_source_results_and_summaries_validate(self):
        id3 = analysis._canonicalize_results(
            pd.read_csv(ROOT / "results/tables/shared_3x5_id3_results.csv"),
            "ID3",
        )
        knn = analysis._canonicalize_results(
            pd.read_csv(
                ROOT
                / "results/tables/shared_3x5_knn_phase6_corrected_results.csv"
            ),
            "k-NN",
        )
        analysis.validate_model_results(id3, "ID3")
        analysis.validate_model_results(knn, "corrected k-NN")
        analysis.validate_source_summary(
            id3,
            pd.read_csv(ROOT / "results/tables/shared_3x5_id3_summary.csv"),
            "ID3",
        )
        analysis.validate_source_summary(
            knn,
            pd.read_csv(
                ROOT
                / "results/tables/shared_3x5_knn_phase6_corrected_summary.csv"
            ),
            "corrected k-NN",
        )

    def test_generated_tables_have_expected_shapes_and_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "tables"
            figure_dir = Path(directory) / "figures"
            generated = analysis.run_analysis(
                output_dir=output_dir,
                figure_dir=figure_dir,
                overwrite=False,
                bootstrap_resamples=200,
            )
            self.assertEqual(
                [len(pd.read_csv(path)) for path in generated["tables"]],
                [120, 8, 60, 4, 60, 4],
            )
            for path in generated["figures"]:
                self.assertTrue(path.is_file())
            leakage = pd.read_csv(generated["tables"][0])
            model = pd.read_csv(generated["tables"][2])
            selector = pd.read_csv(generated["tables"][4])
            self.assertTrue(leakage["global_fold"].between(1, 15).all())
            self.assertTrue(model["global_fold"].between(1, 15).all())
            self.assertTrue(selector["global_fold"].between(1, 15).all())
            self.assertTrue(
                leakage["leaky_minus_safe"].between(-1, 1).all()
            )

    def test_superseded_knn_input_is_rejected(self):
        with self.assertRaises(ValueError):
            analysis.run_analysis(
                knn_results=ROOT / "results/tables/shared_3x5_knn_results.csv",
                knn_summary=ROOT / "results/tables/shared_3x5_knn_summary.csv",
                output_dir=Path(tempfile.mkdtemp()),
                figure_dir=Path(tempfile.mkdtemp()),
            )


if __name__ == "__main__":
    unittest.main()
