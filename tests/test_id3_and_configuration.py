import csv
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_id3_experiment as id3  # noqa: E402
import compare_leakage as knn  # noqa: E402
from experiment_config import (  # noqa: E402
    DEFAULT_CONFIG,
    make_outer_splits,
    outer_split_manifest_rows,
    validate_outer_splits,
)


class Id3CoreTests(unittest.TestCase):
    def test_entropy_and_information_gain(self):
        self.assertAlmostEqual(id3.calculate_entropy([]), 0.0)
        self.assertAlmostEqual(
            id3.calculate_entropy([0, 0, 1, 1]),
            1.0,
        )
        self.assertAlmostEqual(
            id3.calculate_information_gain(
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
            ),
            1.0,
        )
        self.assertAlmostEqual(
            id3.calculate_information_gain(
                np.array([0, 0, 1, 1]),
                np.array([1, 1, 1, 1]),
            ),
            0.0,
        )

    def test_pure_node_is_leaf(self):
        tree = id3.build_id3_tree(
            np.array([[0], [1], [0]]),
            np.array([1, 1, 1]),
            np.array(["feature"]),
        )
        self.assertEqual(tree, {"leaf": 1})

    def test_empty_labels_raise_and_exhausted_features_use_majority(self):
        with self.assertRaises(ValueError):
            id3.build_id3_tree(
                np.empty((0, 1), dtype=int),
                np.array([], dtype=int),
                np.array(["feature"]),
            )

        tree = id3.build_id3_tree(
            np.empty((4, 0), dtype=int),
            np.array([0, 0, 1, 1]),
            np.array([], dtype=object),
        )
        self.assertEqual(tree, {"leaf": 0})

    def test_depth_and_minimum_sample_thresholds_stop_splitting(self):
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        names = np.array(["feature"])

        depth_limited = id3.build_id3_tree(
            X,
            y,
            names,
            max_depth=0,
        )
        min_sample_limited = id3.build_id3_tree(
            X,
            y,
            names,
            min_samples_split=5,
        )

        self.assertEqual(depth_limited, {"leaf": 0})
        self.assertEqual(min_sample_limited, {"leaf": 0})

    def test_tree_and_predictions_are_deterministic(self):
        X = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        y = np.array([0, 0, 1, 1])
        names = np.array(["first", "second"])

        tree_a = id3.build_id3_tree(X, y, names)
        tree_b = id3.build_id3_tree(X, y, names)

        self.assertEqual(tree_a, tree_b)
        np.testing.assert_array_equal(
            id3.predict_batch(tree_a, X, names),
            y,
        )


class LeakageBoundaryTests(unittest.TestCase):
    def _saved_manifest_splits(self):
        with (ROOT / "configs" / "outer_fold_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as manifest_file:
            rows = list(csv.DictReader(manifest_file))

        for row_number, row in enumerate(rows, start=1):
            self.assertEqual(int(row["seed"]), DEFAULT_CONFIG.seed)
            self.assertEqual(
                int(row["outer_splits"]),
                DEFAULT_CONFIG.outer_splits,
            )
            self.assertEqual(
                int(row["outer_repeats"]),
                DEFAULT_CONFIG.outer_repeats,
            )
            self.assertEqual(int(row["global_fold"]), row_number)

        return [
            (
                np.fromstring(row["train_indices"], sep=";", dtype=int),
                np.fromstring(row["test_indices"], sep=";", dtype=int),
            )
            for row in rows
        ]

    def test_id3_safe_selection_uses_training_fold_only(self):
        X_train = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 1.0, 0.0],
                [0.2, 0.0, 1.0],
                [0.3, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 1.0, 1.0],
                [1.2, 0.0, 0.0],
                [1.3, 1.0, 1.0],
            ]
        )
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_test = np.array([[0.15, 0.0, 0.5], [1.15, 1.0, 0.5]])
        feature_names = np.array(["signal", "noise_a", "noise_b"])

        original_n_features = id3.N_FEATURES
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, [0, 1]])
        id3.N_FEATURES = 1
        try:
            first = id3.select_fold_features(
                X_train,
                X_test,
                y_train,
                "ANOVA",
                feature_names,
                X_all=X_all,
                y_all=y_all,
            )
            # Even when held-out labels are available to the caller, the safe
            # condition must ignore them.
            y_all[-2:] = [1, 0]
            second = id3.select_fold_features(
                X_train,
                X_test,
                y_train,
                "ANOVA",
                feature_names,
                X_all=X_all,
                y_all=y_all,
            )
        finally:
            id3.N_FEATURES = original_n_features

        np.testing.assert_array_equal(first[3], [0])
        np.testing.assert_array_equal(first[3], second[3])
        np.testing.assert_array_equal(first[2], ["signal"])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_id3_leaky_selection_requires_all_labels(self):
        with self.assertRaisesRegex(ValueError, "requires all observations"):
            id3.select_fold_features(
                np.zeros((4, 2)),
                np.zeros((2, 2)),
                np.array([0, 0, 1, 1]),
                "ANOVA",
                np.array(["a", "b"]),
                leakage_mode="leaky",
            )

    def test_shared_outer_splits_are_deterministic_and_matched(self):
        labels = np.array([0, 0, 0, 1, 1, 1] * 2)
        first = make_outer_splits(labels)
        second = make_outer_splits(labels)

        self.assertEqual(len(first), DEFAULT_CONFIG.outer_splits * DEFAULT_CONFIG.outer_repeats)
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)

        manifest_rows = outer_split_manifest_rows(labels)
        self.assertEqual(len(manifest_rows), len(first))
        self.assertEqual(
            manifest_rows[0]["test_indices"],
            ";".join(map(str, first[0][1])),
        )
        validate_outer_splits(labels, first)

    def test_outer_split_generation_rejects_too_small_class_before_splitter(self):
        labels = np.array([0, 0, 0, 1, 1])
        with self.assertRaisesRegex(
            ValueError,
            r"Each class must contain at least outer_splits observations.*smallest class has 2",
        ):
            make_outer_splits(labels)

    def test_outer_split_generation_accepts_class_count_equal_to_fold_count(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        splits = make_outer_splits(labels)
        self.assertEqual(len(splits), DEFAULT_CONFIG.outer_splits * DEFAULT_CONFIG.outer_repeats)
        for _, test_idx in splits:
            self.assertEqual(set(labels[test_idx]), {0, 1})

    def test_outer_split_validation_rejects_duplicate_fold_within_repeat(self):
        labels = np.array([0, 0, 0, 1, 1, 1] * 2)
        splits = make_outer_splits(labels)
        malformed = list(splits)
        malformed[1] = splits[0]

        with self.assertRaisesRegex(ValueError, "distinct test folds"):
            validate_outer_splits(labels, malformed)

    def test_outer_split_validation_rejects_missing_fold_within_repeat(self):
        labels = np.array([0, 0, 0, 1, 1, 1] * 2)
        splits = make_outer_splits(labels)
        malformed = list(splits)
        malformed[2] = splits[0]

        with self.assertRaisesRegex(ValueError, "distinct test folds"):
            validate_outer_splits(labels, malformed)

    def test_saved_manifest_is_complete_and_disjoint(self):
        manifest_splits = self._saved_manifest_splits()

        self.assertEqual(len(manifest_splits), 15)
        test_counts = np.zeros(72, dtype=int)
        for train_idx, test_idx in manifest_splits:
            self.assertEqual(len(train_idx), 48)
            self.assertEqual(len(test_idx), 24)
            self.assertEqual(len(np.unique(train_idx)), len(train_idx))
            self.assertEqual(len(np.unique(test_idx)), len(test_idx))
            self.assertEqual(np.intersect1d(train_idx, test_idx).size, 0)
            np.testing.assert_array_equal(
                np.sort(np.concatenate([train_idx, test_idx])),
                np.arange(72),
            )
            test_counts[test_idx] += 1

        np.testing.assert_array_equal(
            test_counts,
            np.full(72, DEFAULT_CONFIG.outer_repeats),
        )

    def test_saved_manifest_can_be_reused_by_both_models(self):
        splits_for_id3 = self._saved_manifest_splits()
        splits_for_knn = self._saved_manifest_splits()

        for id3_split, knn_split in zip(splits_for_id3, splits_for_knn):
            np.testing.assert_array_equal(id3_split[0], knn_split[0])
            np.testing.assert_array_equal(id3_split[1], knn_split[1])

    def test_preprocessing_statistics_are_training_only(self):
        X_train = np.array([[0.0, np.nan], [2.0, 6.0]])
        X_test = np.array([[100.0, 1000.0]])

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_imputed = imputer.fit_transform(X_train)
        scaler.fit(X_train_imputed)

        np.testing.assert_array_equal(imputer.statistics_, [1.0, 6.0])
        np.testing.assert_array_equal(scaler.mean_, [1.0, 6.0])
        np.testing.assert_array_equal(
            scaler.transform(imputer.transform(X_test)),
            np.array([[99.0, 994.0]]),
        )

    def test_knn_preprocessing_is_shared_and_training_only(self):
        X_train = np.array(
            [[0.0, np.nan], [2.0, 6.0]],
        )
        X_test = np.array([[100.0, 1000.0]])
        X_all = np.vstack([X_train, X_test])

        (
            X_train_scaled,
            X_test_scaled,
            X_all_scaled,
            imputer,
            scaler,
        ) = knn.preprocess_fold(X_train, X_test, X_all)

        np.testing.assert_array_equal(imputer.statistics_, [1.0, 6.0])
        np.testing.assert_array_equal(scaler.mean_, [1.0, 6.0])
        np.testing.assert_array_equal(
            X_test_scaled,
            scaler.transform(imputer.transform(X_test)),
        )
        np.testing.assert_array_equal(
            X_all_scaled,
            scaler.transform(imputer.transform(X_all)),
        )
        self.assertEqual(X_train_scaled.shape[0], len(X_train))

    def test_knn_conditions_differ_only_in_selection_labels(self):
        X_train = np.array(
            [[0.0, 0.0], [0.1, 1.0], [1.0, 0.0], [1.1, 1.0]],
        )
        X_test = np.array([[0.2, 0.0], [0.9, 1.0]])
        X_all = np.vstack([X_train, X_test])
        y_train = np.array([0, 0, 1, 1])
        y_all = np.concatenate([y_train, [1, 0]])
        (
            X_train_scaled,
            X_test_scaled,
            X_all_scaled,
            _imputer,
            _scaler,
        ) = knn.preprocess_fold(X_train, X_test, X_all)

        selector_labels = []
        classifier_fit_rows = []
        classifier_predict_rows = []

        class RecordingSelector:
            def fit(self, features, labels):
                selector_labels.append(np.array(labels, copy=True))
                self._indices = np.array([0])
                return self

            def get_support(self, indices=False):
                if indices:
                    return self._indices
                return np.array([True, False])

        class RecordingClassifier:
            def fit(self, features, labels):
                classifier_fit_rows.append((features.copy(), labels.copy()))
                return self

            def predict(self, features):
                classifier_predict_rows.append(features.copy())
                return np.zeros(features.shape[0], dtype=int)

        original_selector_factory = knn.make_selector
        original_classifier_factory = knn.make_knn_classifier
        knn.make_selector = lambda method: RecordingSelector()
        knn.make_knn_classifier = lambda: RecordingClassifier()
        try:
            safe_predictions, safe_indices = knn.fit_knn_condition(
                X_train_scaled,
                X_test_scaled,
                y_train,
                X_train_scaled,
                y_train,
                "anova",
            )
            leaky_predictions, leaky_indices = knn.fit_knn_condition(
                X_train_scaled,
                X_test_scaled,
                y_train,
                X_all_scaled,
                y_all,
                "anova",
            )
        finally:
            knn.make_selector = original_selector_factory
            knn.make_knn_classifier = original_classifier_factory

        np.testing.assert_array_equal(selector_labels[0], y_train)
        np.testing.assert_array_equal(selector_labels[1], y_all)
        np.testing.assert_array_equal(safe_indices, leaky_indices)
        self.assertEqual(len(safe_predictions), len(X_test))
        self.assertEqual(len(leaky_predictions), len(X_test))
        self.assertEqual(len(classifier_fit_rows), 2)
        self.assertTrue(
            all(len(labels) == len(y_train) for _, labels in classifier_fit_rows)
        )
        self.assertTrue(
            all(
                len(features) == len(X_test)
                for features in classifier_predict_rows
            )
        )

    def test_knn_classifier_settings_are_shared_without_second_scaler(self):
        classifier = knn.make_knn_classifier()

        self.assertIsInstance(classifier, knn.KNeighborsClassifier)
        self.assertEqual(
            classifier.n_neighbors,
            DEFAULT_CONFIG.knn_neighbors,
        )
        self.assertEqual(knn.K_FEATURES, DEFAULT_CONFIG.selected_features)


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_modify_existing_result_files(self):
        result_files = sorted((ROOT / "results" / "tables").glob("*.csv"))
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in result_files
        }

        self.assertTrue(callable(id3.main))
        self.assertTrue(callable(id3.build_id3_tree))

        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in result_files
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
