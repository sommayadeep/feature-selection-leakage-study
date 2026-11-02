import csv
import hashlib
import io
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from unittest import mock

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import measure_feature_stability as stability  # noqa: E402


class FeatureStabilityTests(unittest.TestCase):
    def test_dataset_hash_is_deterministic_and_tracks_exact_bytes(self):
        first = b"target,g1,g2\nA,1,2\nB,3,4\n"
        same = b"target,g1,g2\nA,1,2\nB,3,4\n"
        changed = b"target,g1,g2\nA,1,2\nB,3,5\n"

        self.assertEqual(
            stability.dataset_sha256(first), stability.dataset_sha256(same)
        )
        self.assertNotEqual(
            stability.dataset_sha256(first), stability.dataset_sha256(changed)
        )
        self.assertEqual(stability.dataset_sha256(first), hashlib.sha256(first).hexdigest())

    def test_dataset_hash_changes_with_sample_feature_or_label_order_and_values(self):
        baseline = b"target,g1,g2\nA,1,2\nB,3,4\n"
        variants = (
            b"target,g1,g2\nB,3,4\nA,1,2\n",
            b"target,g2,g1\nA,2,1\nB,4,3\n",
            b"target,g1,g2\nB,1,2\nA,3,4\n",
        )
        baseline_hash = stability.dataset_sha256(baseline)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(baseline_hash, stability.dataset_sha256(variant))

    def test_split_manifest_requires_a_valid_sha256_digest(self):
        labels = [0] * 12 + [1] * 8
        splits = stability.make_stability_splits(labels, n_repeats=2)
        for invalid in (None, "abc", "g" * 64, "A" * 64):
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with self.assertRaisesRegex(ValueError, "64-character"):
                        stability.write_stability_split_manifest(
                            Path(temp_dir) / "manifest.csv", labels, splits,
                            n_repeats=2, dataset_hash=invalid,
                        )

    def test_manifest_fingerprints_all_twenty_splits_and_experiment_parses_hashed_bytes(self):
        content = b"target,g1,g2\n" + b"".join(
            f"{'A' if i < 12 else 'B'},{i},{i + 1}\n".encode()
            for i in range(20)
        )
        expected_hash = hashlib.sha256(content).hexdigest()
        parsed = pd.read_csv(io.BytesIO(content))
        encoded = LabelEncoder().fit_transform(parsed["target"].astype(str))
        splits = stability.make_stability_splits(encoded, n_repeats=20)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.csv"
            output = Path(temp_dir) / "out"
            source.write_bytes(content)
            with (
                mock.patch.object(
                    stability,
                    "SPLIT_MANIFEST_FILE",
                    Path(temp_dir) / "manifest.csv",
                ),
                mock.patch.object(
                    stability,
                    "select_genes",
                    side_effect=[{"g1"} for _ in range(40)],
                ),
            ):
                stability.run_experiment(source, output, n_repeats=20, n_features=1)
            with (Path(temp_dir) / "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as manifest_file:
                rows = list(csv.DictReader(manifest_file))

        self.assertEqual(len(rows), 20)
        self.assertEqual({row["dataset_sha256"] for row in rows}, {expected_hash})

    def test_import_exposes_calculations_without_loading_experiment_data(self):
        self.assertTrue(callable(stability.jaccard_similarity))
        self.assertTrue(callable(stability.summarize_stability))
        with mock.patch(
            "pandas.read_csv",
            side_effect=AssertionError("module import must not load experiment data"),
        ):
            importlib.reload(stability)

    def test_jaccard_similarity_handles_overlap_and_empty_sets(self):
        self.assertAlmostEqual(
            stability.jaccard_similarity({"a", "b"}, {"b", "c"}),
            1 / 3,
        )
        self.assertEqual(stability.jaccard_similarity(set(), set()), 1.0)

    def test_summary_reports_all_unordered_pairs_and_sample_standard_deviation(self):
        selected = {
            "anova": [
                {"a", "b"},
                {"a", "c"},
                {"b", "c"},
            ]
        }
        summary = stability.summarize_stability(
            selected,
            repeats=3,
            features_per_repeat=2,
        )[0]

        self.assertEqual(summary["number_of_pairs"], 3)
        self.assertAlmostEqual(summary["mean_jaccard"], 1 / 3)
        self.assertAlmostEqual(summary["std_jaccard"], 0.0)

    def test_summary_rejects_unexpected_selection_size(self):
        with self.assertRaisesRegex(ValueError, "exactly 2 features"):
            stability.summarize_stability(
                {"anova": [{"a", "b"}, {"a"}]},
                repeats=2,
                features_per_repeat=2,
            )

    def test_summary_requires_two_resamples_for_sample_standard_deviation(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            stability.summarize_stability(
                {"anova": [{"a", "b"}]},
                repeats=1,
                features_per_repeat=2,
            )

    def test_stability_splits_are_archived_with_reconstructable_indices(self):
        labels = [0] * 12 + [1] * 8
        splits = stability.make_stability_splits(
            labels,
            n_repeats=20,
            test_size=0.25,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "stability_splits.csv"
            stability.write_stability_split_manifest(
                manifest_path,
                labels,
                splits,
                n_repeats=20,
                test_size=0.25,
                seed=42,
                dataset_hash="a" * 64,
            )
            with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
                rows = list(csv.DictReader(manifest_file))

        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["seed"], "42")
        self.assertEqual(rows[0]["n_repeats"], "20")
        self.assertEqual(rows[0]["test_size"], "0.25")
        for row, (expected_train, expected_test) in zip(rows, splits):
            np.testing.assert_array_equal(
                np.fromstring(row["train_indices"], sep=";", dtype=int),
                expected_train,
            )
            np.testing.assert_array_equal(
                np.fromstring(row["test_indices"], sep=";", dtype=int),
                expected_test,
            )
            self.assertEqual(int(row["train_size"]), len(expected_train))
            self.assertEqual(int(row["test_samples"]), len(expected_test))

    def test_stability_split_manifest_rejects_a_changed_split_schedule(self):
        labels = [0] * 12 + [1] * 8
        splits = stability.make_stability_splits(labels, n_repeats=2)
        altered = list(splits)
        altered[0] = (splits[0][1], splits[0][0])

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError,
                "differs from the configured schedule",
            ):
                stability.write_stability_split_manifest(
                    Path(temp_dir) / "stability_splits.csv",
                    labels,
                    altered,
                    n_repeats=2,
                    dataset_hash="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
