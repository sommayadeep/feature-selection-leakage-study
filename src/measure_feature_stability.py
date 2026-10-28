"""Measure feature-selection stability across stratified training resamples."""

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

N_REPEATS = 20
N_FEATURES = 100
TEST_SIZE = 0.25
SEED = 42


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

    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=seed,
    )
    selected_sets = {"anova": [], "mutual_information": []}

    print("=" * 60)
    print("FEATURE-SELECTION STABILITY EXPERIMENT")
    print("=" * 60)
    print(f"Patients: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print(f"Genes selected per repeat: {n_features}")
    print(f"Repeats: {n_repeats}")

    for repeat, (train_idx, _) in enumerate(splitter.split(X, y), start=1):
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
