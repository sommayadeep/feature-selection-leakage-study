"""Shared configuration and split utilities for future comparable experiments."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


@dataclass(frozen=True)
class SharedExperimentConfig:
    """Configuration for the future matched ID3/k-NN outer evaluation."""

    seed: int = 42
    outer_splits: int = 3
    outer_repeats: int = 5
    selected_features: int = 50
    discretization_bins: int = 3
    id3_max_depth: int = 5
    id3_min_samples_split: int = 4
    knn_neighbors: int = 3


DEFAULT_CONFIG = SharedExperimentConfig()


def make_outer_splits(
    labels: np.ndarray,
    config: SharedExperimentConfig = DEFAULT_CONFIG,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified outer splits for matched models."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if len(labels) == 0:
        raise ValueError("labels must not be empty.")
    if config.outer_splits < 2:
        raise ValueError("outer_splits must be at least 2.")
    if config.outer_repeats < 1:
        raise ValueError("outer_repeats must be at least 1.")

    splitter = RepeatedStratifiedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        random_state=config.seed,
    )
    placeholder_features = np.zeros((len(labels), 1), dtype=float)
    return list(splitter.split(placeholder_features, labels))


def validate_outer_splits(
    labels: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    config: SharedExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Validate disjoint, complete, stratified repeated outer splits."""

    labels = np.asarray(labels)
    split_list = list(splits)
    expected_count = config.outer_splits * config.outer_repeats

    if len(split_list) != expected_count:
        raise ValueError(
            f"Expected {expected_count} outer splits, found {len(split_list)}."
        )

    expected_test_counts = np.zeros(len(labels), dtype=int)
    repeat_test_indices: list[list[np.ndarray]] = [
        [] for _ in range(config.outer_repeats)
    ]
    class_labels, class_counts = np.unique(labels, return_counts=True)
    if len(class_labels) < 2:
        raise ValueError("At least two classes are required for stratification.")

    for split_number, (train_idx, test_idx) in enumerate(split_list):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        if (
            np.any(train_idx < 0)
            or np.any(test_idx < 0)
            or np.any(train_idx >= len(labels))
            or np.any(test_idx >= len(labels))
        ):
            raise ValueError("A split contains an out-of-range index.")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError("Training and test indices overlap.")
        if not np.array_equal(
            np.union1d(train_idx, test_idx),
            np.arange(len(labels)),
        ):
            raise ValueError("A split does not cover every patient.")
        if len(np.unique(test_idx)) != len(test_idx):
            raise ValueError("A test split contains duplicate indices.")

        train_class_counts = np.array(
            [np.sum(labels[train_idx] == label) for label in class_labels]
        )
        test_class_counts = np.array(
            [np.sum(labels[test_idx] == label) for label in class_labels]
        )
        if np.any(train_class_counts == 0) or np.any(test_class_counts == 0):
            raise ValueError("Every split must contain every class.")

        # StratifiedKFold distributes each class across folds as evenly as
        # integer fold sizes permit.
        expected_test_size = class_counts // config.outer_splits
        if any(
            int(count) not in {int(base), int(base + 1)}
            for count, base in zip(test_class_counts, expected_test_size)
        ):
            raise ValueError("A test split is not class-stratified.")

        expected_test_counts[test_idx] += 1
        repeat_number = split_number // config.outer_splits
        repeat_test_indices[repeat_number].append(test_idx)

    for repeat_number, test_folds in enumerate(repeat_test_indices, start=1):
        if len({tuple(sorted(test_idx)) for test_idx in test_folds}) != config.outer_splits:
            raise ValueError(
                f"Repeat {repeat_number} must contain {config.outer_splits} "
                "distinct test folds."
            )

    if not np.all(expected_test_counts == config.outer_repeats):
        raise ValueError(
            "Each patient must appear in the test set once per repeat."
        )


def load_outer_split_manifest(
    manifest_path: str | Path,
    labels: np.ndarray,
    config: SharedExperimentConfig = DEFAULT_CONFIG,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load and validate the deterministic shared outer-fold manifest."""

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Outer-fold manifest not found: {manifest_path}")

    labels = np.asarray(labels)
    expected_splits = make_outer_splits(labels, config)
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if len(rows) != len(expected_splits):
        raise ValueError("Manifest row count does not match shared configuration.")

    loaded_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for row_number, (row, expected) in enumerate(
        zip(rows, expected_splits),
        start=1,
    ):
        for field in ("seed", "outer_splits", "outer_repeats"):
            if int(row[field]) != getattr(config, field):
                raise ValueError(
                    f"Manifest row {row_number} has inconsistent {field}."
                )

        train_idx = np.fromstring(row["train_indices"], sep=";", dtype=int)
        test_idx = np.fromstring(row["test_indices"], sep=";", dtype=int)
        if not np.array_equal(train_idx, expected[0]):
            raise ValueError(f"Manifest train indices differ at row {row_number}.")
        if not np.array_equal(test_idx, expected[1]):
            raise ValueError(f"Manifest test indices differ at row {row_number}.")
        if int(row["train_size"]) != len(train_idx):
            raise ValueError(f"Manifest train size differs at row {row_number}.")
        if int(row["test_size"]) != len(test_idx):
            raise ValueError(f"Manifest test size differs at row {row_number}.")
        loaded_splits.append((train_idx, test_idx))

    validate_outer_splits(labels, loaded_splits, config)
    return loaded_splits


def outer_split_manifest_rows(
    labels: np.ndarray,
    config: SharedExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, object]]:
    """Serialize the shared outer splits into stable, auditable row records."""

    labels = np.asarray(labels)
    splits = make_outer_splits(labels, config)
    rows: list[dict[str, object]] = []

    for global_fold, (train_idx, test_idx) in enumerate(splits, start=1):
        repeat = (global_fold - 1) // config.outer_splits + 1
        fold = (global_fold - 1) % config.outer_splits + 1
        rows.append(
            {
                "seed": config.seed,
                "outer_splits": config.outer_splits,
                "outer_repeats": config.outer_repeats,
                "repeat": repeat,
                "fold": fold,
                "global_fold": global_fold,
                "train_indices": ";".join(map(str, train_idx)),
                "test_indices": ";".join(map(str, test_idx)),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            }
        )

    return rows
