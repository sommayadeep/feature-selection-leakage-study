"""Measure feature-selection stability across stratified training resamples."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "leukemia.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
SPLIT_MANIFEST_FILE = (
    PROJECT_ROOT / "configs" / "feature_stability_split_manifest.csv"
)

N_REPEATS = 20
N_FEATURES = 100
TEST_SIZE = 0.25
SEED = 42


def make_stability_splits(
    labels,
    n_repeats=N_REPEATS,
    test_size=TEST_SIZE,
    seed=SEED,
):
    """Return the existing seeded stratified holdout schedule as index arrays."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=seed,
    )
    placeholder_features = np.zeros((len(labels), 1), dtype=float)
    return list(splitter.split(placeholder_features, labels))


def write_stability_split_manifest(
    manifest_path,
    labels,
    splits,
    n_repeats=N_REPEATS,
    test_size=TEST_SIZE,
    seed=SEED,
):
    """Archive exact sample indices and protocol settings for every resample."""

    labels = np.asarray(labels)
    splits = list(splits)
    if len(splits) != n_repeats:
        raise ValueError(f"Expected {n_repeats} resamples, found {len(splits)}.")

    expected = make_stability_splits(labels, n_repeats, test_size, seed)
    rows = []
    for repeat, ((train_idx, test_idx), (expected_train, expected_test)) in enumerate(
        zip(splits, expected),
        start=1,
    ):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        if not np.array_equal(train_idx, expected_train) or not np.array_equal(
            test_idx, expected_test
        ):
            raise ValueError(f"Resample {repeat} differs from the configured schedule.")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"Resample {repeat} has overlapping train/test indices.")
        if not np.array_equal(
            np.sort(np.concatenate((train_idx, test_idx))), np.arange(len(labels))
        ):
            raise ValueError(
                f"Resample {repeat} does not cover all samples exactly once."
            )
        rows.append(
            {
                "seed": seed,
                "n_repeats": n_repeats,
                "test_size": test_size,
                "repeat": repeat,
                "train_size": len(train_idx),
                "test_samples": len(test_idx),
                "train_indices": ";".join(map(str, train_idx)),
                "test_indices": ";".join(map(str, test_idx)),
            }
        )

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def select_genes(
    X_train,
    y_train,
    gene_names,
    method,
    n_features=N_FEATURES,
    seed=SEED,
):
    """Fit imputation and feature selection on one training subset only."""

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)

    if method == "anova":
        selector = SelectKBest(score_func=f_classif, k=n_features)
    elif method == "mutual_information":
        selector = SelectKBest(
            score_func=lambda features, labels: mutual_info_classif(
                features,
                labels,
                random_state=seed,
            ),
            k=n_features,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    selector.fit(X_imputed, y_train)
    selected_indices = selector.get_support(indices=True)
    return set(np.asarray(gene_names)[selected_indices])


def jaccard_similarity(set_a, set_b):
    """Return intersection-over-union for two selected-feature sets."""

    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def summarize_stability(
    selected_sets,
    repeats=N_REPEATS,
    features_per_repeat=N_FEATURES,
):
    """Return the existing per-selector pairwise Jaccard summary records."""

    if repeats < 2:
        raise ValueError("At least two resampled selections are required.")
    rows = []
    for method, gene_sets in selected_sets.items():
        if len(gene_sets) != repeats:
            raise ValueError(
                f"{method} must contain exactly {repeats} resampled selections."
            )
        if any(len(gene_set) != features_per_repeat for gene_set in gene_sets):
            raise ValueError(
                f"Each {method} selection must contain exactly "
                f"{features_per_repeat} features."
            )

        pairwise_scores = [
            jaccard_similarity(gene_sets[i], gene_sets[j])
            for i in range(len(gene_sets))
            for j in range(i + 1, len(gene_sets))
        ]
        rows.append(
            {
                "selector": method,
                "repeats": repeats,
                "features_per_repeat": features_per_repeat,
                "number_of_pairs": len(pairwise_scores),
                "mean_jaccard": float(np.mean(pairwise_scores)),
                "std_jaccard": float(np.std(pairwise_scores, ddof=1)),
            }
        )
    return rows


def run_experiment(
    data_file=DATA_FILE,
    output_dir=OUTPUT_DIR,
    n_repeats=N_REPEATS,
    n_features=N_FEATURES,
    test_size=TEST_SIZE,
    seed=SEED,
):
    """Run the configured stability experiment and save its summary CSV."""

    data = pd.read_csv(data_file)
    X = data.drop(columns="target").to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(data["target"].astype(str))
    gene_names = data.drop(columns="target").columns.to_numpy()

    splits = make_stability_splits(y, n_repeats, test_size, seed)
    write_stability_split_manifest(
        SPLIT_MANIFEST_FILE,
        y,
        splits,
        n_repeats=n_repeats,
        test_size=test_size,
        seed=seed,
    )
    selected_sets = {"anova": [], "mutual_information": []}

    print("=" * 60)
    print("FEATURE-SELECTION STABILITY EXPERIMENT")
    print("=" * 60)
    print(f"Patients: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print(f"Genes selected per repeat: {n_features}")
    print(f"Repeats: {n_repeats}")

    for repeat, (train_idx, _) in enumerate(splits, start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        for method in selected_sets:
            selected_sets[method].append(
                select_genes(
                    X_train,
                    y_train,
                    gene_names,
                    method,
                    n_features=n_features,
                    seed=seed,
                )
            )
        print(f"Completed repeat {repeat}/{n_repeats}")

    stability_df = pd.DataFrame(
        summarize_stability(
            selected_sets,
            repeats=n_repeats,
            features_per_repeat=n_features,
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "feature_stability.csv"
    stability_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("FEATURE-SELECTION STABILITY SUMMARY")
    print("=" * 60)
    print(stability_df.to_string(index=False))
    print(f"\nResults saved to:\n{output_file}")
    print(
        "\nInterpretation: a higher mean Jaccard score means "
        "the selected gene sets overlap more across training subsets."
    )
    return stability_df


def main():
    run_experiment()


if __name__ == "__main__":
    main()
