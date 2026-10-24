from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "leukemia.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

N_REPEATS = 20
N_FEATURES = 100
TEST_SIZE = 0.25
SEED = 42


# =====================================================
# 2. LOAD DATA
# =====================================================

data = pd.read_csv(DATA_FILE)

X = data.drop(columns="target").to_numpy(dtype=float)
y = LabelEncoder().fit_transform(data["target"].astype(str))

gene_names = data.drop(columns="target").columns.to_numpy()

print("=" * 60)
print("FEATURE-SELECTION STABILITY EXPERIMENT")
print("=" * 60)

print(f"Patients: {X.shape[0]}")
print(f"Genes: {X.shape[1]}")
print(f"Genes selected per repeat: {N_FEATURES}")
print(f"Repeats: {N_REPEATS}")


# =====================================================
# 3. DEFINE FEATURE-SELECTION METHODS
# =====================================================

def select_genes(X_train, y_train, method):

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)

    if method == "anova":
        selector = SelectKBest(
            score_func=f_classif,
            k=N_FEATURES,
        )

    elif method == "mutual_information":
        selector = SelectKBest(
            score_func=lambda features, labels: mutual_info_classif(
                features,
                labels,
                random_state=SEED,
            ),
            k=N_FEATURES,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    selector.fit(X_imputed, y_train)

    selected_indices = selector.get_support(indices=True)

    return set(gene_names[selected_indices])


# =====================================================
# 4. GENERATE TRAINING SUBSETS
# =====================================================

splitter = StratifiedShuffleSplit(
    n_splits=N_REPEATS,
    test_size=TEST_SIZE,
    random_state=SEED,
)

selected_sets = {
    "anova": [],
    "mutual_information": [],
}

for repeat, (train_idx, test_idx) in enumerate(
    splitter.split(X, y),
    start=1,
):

    X_train = X[train_idx]
    y_train = y[train_idx]

    for method in selected_sets:

        selected = select_genes(
            X_train,
            y_train,
            method,
        )

        selected_sets[method].append(selected)

    print(f"Completed repeat {repeat}/{N_REPEATS}")


# =====================================================
# 5. CALCULATE JACCARD SIMILARITY
# =====================================================

def jaccard_similarity(set_a, set_b):

    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


stability_rows = []

for method, gene_sets in selected_sets.items():

    pairwise_scores = []

    for i in range(len(gene_sets)):
        for j in range(i + 1, len(gene_sets)):

            score = jaccard_similarity(
                gene_sets[i],
                gene_sets[j],
            )

            pairwise_scores.append(score)

    stability_rows.append({
        "selector": method,
        "repeats": N_REPEATS,
        "features_per_repeat": N_FEATURES,
        "number_of_pairs": len(pairwise_scores),
        "mean_jaccard": float(np.mean(pairwise_scores)),
        "std_jaccard": float(np.std(pairwise_scores, ddof=1)),
    })


# =====================================================
# 6. SAVE RESULTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stability_df = pd.DataFrame(stability_rows)

output_file = OUTPUT_DIR / "feature_stability.csv"

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