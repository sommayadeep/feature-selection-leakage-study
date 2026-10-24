from pathlib import Path
from collections import Counter
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

DATA_FILE = ROOT / "data" / "processed" / "leukemia.csv"
RESULTS_FILE = TABLES / "id3_experiment_results.csv"
GENES_FILE = TABLES / "id3_feature_selection_records.csv"

SEED = 42
N_SPLITS = 3
N_REPEATS = 5
N_FEATURES = 50
N_BINS = 3
MAX_DEPTH = 5
MIN_SAMPLES_SPLIT = 4

TARGET_REPEAT = 3
TARGET_FOLD = 2


# ============================================================
# 2. ORIGINAL ID3 FUNCTIONS
# These reproduce the logic in run_id3_experiment.py.
# ============================================================

def calculate_entropy(y):
    y = np.asarray(y)

    if len(y) == 0:
        return 0.0

    counts = Counter(y)
    total = len(y)
    result = 0.0

    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return float(result)


def calculate_information_gain(y, feature_values):
    y = np.asarray(y)
    feature_values = np.asarray(feature_values)

    if len(y) != len(feature_values):
        raise ValueError("Labels and feature values must have equal lengths.")

    if len(y) == 0:
        return 0.0

    parent_entropy = calculate_entropy(y)
    weighted_child_entropy = 0.0

    for value in np.unique(feature_values):
        mask = feature_values == value
        child_labels = y[mask]
        proportion = len(child_labels) / len(y)

        weighted_child_entropy += (
            proportion * calculate_entropy(child_labels)
        )

    return max(0.0, float(parent_entropy - weighted_child_entropy))


def majority_class(y):
    if len(y) == 0:
        raise ValueError("Cannot find the majority class of empty labels.")

    return Counter(y).most_common(1)[0][0]


def build_id3_tree(
    X,
    y,
    feature_names,
    depth=0,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
):
    X = np.asarray(X)
    y = np.asarray(y)
    feature_names = np.asarray(feature_names, dtype=object)

    if len(y) == 0:
        raise ValueError("Cannot build a tree from an empty dataset.")

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")

    if X.shape[0] != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if X.shape[1] != len(feature_names):
        raise ValueError("Feature names must match the columns of X.")

    if len(np.unique(y)) == 1:
        return {"leaf": y[0]}

    if depth >= max_depth:
        return {"leaf": majority_class(y)}

    if len(y) < min_samples_split:
        return {"leaf": majority_class(y)}

    if X.shape[1] == 0:
        return {"leaf": majority_class(y)}

    gains = np.array([
        calculate_information_gain(y, X[:, j])
        for j in range(X.shape[1])
    ])

    best_index = int(np.argmax(gains))
    best_gain = float(gains[best_index])

    if best_gain <= 1e-12:
        return {"leaf": majority_class(y)}

    best_feature_name = feature_names[best_index]

    tree = {
        "feature": best_feature_name,
        "majority": majority_class(y),
        "gain": best_gain,
        "branches": {},
    }

    remaining_features = np.delete(feature_names, best_index)
    remaining_X = np.delete(X, best_index, axis=1)

    for value in np.unique(X[:, best_index]):
        mask = X[:, best_index] == value

        child_tree = build_id3_tree(
            remaining_X[mask],
            y[mask],
            remaining_features,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        branch_value = value.item() if hasattr(value, "item") else value
        tree["branches"][branch_value] = child_tree

    return tree


def predict_one(tree, sample, feature_names):
    if "leaf" in tree:
        return tree["leaf"]

    feature_names = list(feature_names)
    feature_index = feature_names.index(tree["feature"])
    feature_value = sample[feature_index]

    if feature_value in tree["branches"]:
        child_tree = tree["branches"][feature_value]

        remaining_sample = np.delete(np.asarray(sample), feature_index)
        remaining_features = np.delete(
            np.asarray(feature_names, dtype=object),
            feature_index,
        )

        return predict_one(
            child_tree,
            remaining_sample,
            remaining_features,
        )

    return tree["majority"]


def predict_batch(tree, X, feature_names):
    return np.asarray([
        predict_one(tree, row, feature_names)
        for row in X
    ])


# ============================================================
# 3. LOAD DATA AND SAVED RESULTS
# ============================================================

data = pd.read_csv(DATA_FILE)

X = data.drop(columns="target").to_numpy(dtype=float)
gene_names = data.drop(columns="target").columns.to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["target"].astype(str).to_numpy())

saved_results = pd.read_csv(RESULTS_FILE)
saved_genes = pd.read_csv(GENES_FILE)

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=SEED,
)

splits = list(cv.split(X, y))

global_fold = (TARGET_REPEAT - 1) * N_SPLITS + TARGET_FOLD
train_idx, test_idx = splits[global_fold - 1]

X_train_raw = X[train_idx]
X_test_raw = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print("=" * 70)
print("STANDALONE ID3 PATIENT-LEVEL DIAGNOSTIC")
print("=" * 70)
print(f"Dataset patients: {len(data)}")
print(f"Training patients: {len(train_idx)}")
print(f"Test patients: {len(test_idx)}")
print(f"Repeat: {TARGET_REPEAT}, fold: {TARGET_FOLD}")
print(f"Global fold: {global_fold}")
print(f"Class labels: {dict(enumerate(label_encoder.classes_))}")


# ============================================================
# 4. IMPUTE USING TRAINING DATA ONLY
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)
X_all_imputed = imputer.transform(X)


# ============================================================
# 5. RUN SAFE AND LEAKY SELECTION
# ============================================================

predictions_by_mode = {}
selected_genes_by_mode = {}
scores_by_mode = {}

for mode in ["safe", "leaky"]:

    selector = SelectKBest(
        score_func=lambda features, labels: mutual_info_classif(
            features,
            labels,
            random_state=SEED,
        ),
        k=N_FEATURES,
    )

    if mode == "safe":
        X_train_selected = selector.fit_transform(
            X_train_imputed, y_train
        )
        X_test_selected = selector.transform(X_test_imputed)

    else:
        # Intentional leakage: all patient labels influence selection.
        selector.fit(X_all_imputed, y)

        X_train_selected = selector.transform(X_train_imputed)
        X_test_selected = selector.transform(X_test_imputed)

    selected_indices = selector.get_support(indices=True)
    selected_genes = gene_names[selected_indices]

    selected_genes_by_mode[mode] = selected_genes

    # Verify the exact gene list against the original saved record.
    saved_gene_row = saved_genes[
        (saved_genes["selector"] == "MI")
        & (saved_genes["selection_mode"] == mode)
        & (saved_genes["repeat"] == TARGET_REPEAT)
        & (saved_genes["fold"] == TARGET_FOLD)
    ]

    if len(saved_gene_row) != 1:
        raise ValueError(
            f"Expected one saved gene record for {mode}; "
            f"found {len(saved_gene_row)}."
        )

    original_gene_list = saved_gene_row.iloc[0]["selected_genes"].split(";")

    genes_match = set(selected_genes) == set(original_gene_list)

    print(f"\n--- {mode.upper()} MI ---")
    print(f"Selected genes: {len(selected_genes)}")
    print(f"Exact saved gene set matches: {genes_match}")

    if not genes_match:
        print("Missing genes:", sorted(set(original_gene_list) - set(selected_genes)))
        print("Unexpected genes:", sorted(set(selected_genes) - set(original_gene_list)))

    discretizer = KBinsDiscretizer(
        n_bins=N_BINS,
        encode="ordinal",
        strategy="quantile",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        X_train_discrete = discretizer.fit_transform(
            X_train_selected
        ).astype(int)

        X_test_discrete = discretizer.transform(
            X_test_selected
        ).astype(int)

    tree = build_id3_tree(
        X_train_discrete,
        y_train,
        selected_genes,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    predictions = predict_batch(
        tree,
        X_test_discrete,
        selected_genes,
    )

    predictions_by_mode[mode] = predictions

    reproduced_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    reproduced_balanced = balanced_accuracy_score(y_test, predictions)

    saved_row = saved_results[
        (saved_results["selector"] == "MI")
        & (saved_results["selection_mode"] == mode)
        & (saved_results["repeat"] == TARGET_REPEAT)
        & (saved_results["fold"] == TARGET_FOLD)
    ]

    if len(saved_row) != 1:
        raise ValueError(
            f"Expected one saved score record for {mode}; "
            f"found {len(saved_row)}."
        )

    original_f1 = float(saved_row.iloc[0]["macro_f1"])
    original_balanced = float(saved_row.iloc[0]["balanced_accuracy"])

    f1_matches = np.isclose(reproduced_f1, original_f1)
    balanced_matches = np.isclose(reproduced_balanced, original_balanced)

    scores_by_mode[mode] = {
        "macro_f1": reproduced_f1,
        "balanced_accuracy": reproduced_balanced,
        "genes_match": genes_match,
        "f1_matches": f1_matches,
        "balanced_matches": balanced_matches,
    }

    print(f"Original Macro-F1:       {original_f1:.4f}")
    print(f"Reproduced Macro-F1:     {reproduced_f1:.4f}")
    print(f"Original balanced acc.:  {original_balanced:.4f}")
    print(f"Reproduced balanced acc.: {reproduced_balanced:.4f}")
    print(f"Macro-F1 matches:        {f1_matches}")
    print(f"Balanced accuracy match: {balanced_matches}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(len(label_encoder.classes_)),
        )
    )


# ============================================================
# 6. COMPARE PATIENT-LEVEL PREDICTIONS
# ============================================================

safe_predictions = predictions_by_mode["safe"]
leaky_predictions = predictions_by_mode["leaky"]

patient_table = pd.DataFrame({
    "patient_row_index": test_idx,
    "true_class": label_encoder.inverse_transform(y_test),
    "safe_prediction": label_encoder.inverse_transform(
        safe_predictions.astype(int)
    ),
    "leaky_prediction": label_encoder.inverse_transform(
        leaky_predictions.astype(int)
    ),
})

patient_table["safe_correct"] = (
    patient_table["true_class"] == patient_table["safe_prediction"]
)

patient_table["leaky_correct"] = (
    patient_table["true_class"] == patient_table["leaky_prediction"]
)

patient_table["prediction_changed"] = (
    patient_table["safe_prediction"] != patient_table["leaky_prediction"]
)

patient_table["safe_wrong_leaky_correct"] = (
    ~patient_table["safe_correct"] & patient_table["leaky_correct"]
)

patient_table["safe_correct_leaky_wrong"] = (
    patient_table["safe_correct"] & ~patient_table["leaky_correct"]
)

patient_table["both_correct"] = (
    patient_table["safe_correct"] & patient_table["leaky_correct"]
)

patient_table["both_wrong"] = (
    ~patient_table["safe_correct"] & ~patient_table["leaky_correct"]
)

print("\n" + "=" * 70)
print("PATIENT-LEVEL COMPARISON")
print("=" * 70)

print(
    "Predictions changed:",
    int(patient_table["prediction_changed"].sum()),
    "of",
    len(patient_table),
)

print(
    "Safe wrong, leaky correct:",
    int(patient_table["safe_wrong_leaky_correct"].sum()),
)

print(
    "Safe correct, leaky wrong:",
    int(patient_table["safe_correct_leaky_wrong"].sum()),
)

print(
    "Both correct:",
    int(patient_table["both_correct"].sum()),
)

print(
    "Both wrong:",
    int(patient_table["both_wrong"].sum()),
)

print("\nPatients whose predictions changed:")
changed = patient_table[patient_table["prediction_changed"]]

if changed.empty:
    print("No predictions changed.")
else:
    print(changed.to_string(index=False))


# ============================================================
# 7. SAVE DIAGNOSTIC OUTPUTS
# ============================================================

patient_output = TABLES / "id3_mi_fold_standalone_patient_comparison.csv"
summary_output = TABLES / "id3_mi_fold_standalone_summary.csv"

patient_table.to_csv(patient_output, index=False)

summary_rows = []

for mode, scores in scores_by_mode.items():
    summary_rows.append({
        "selector": "MI",
        "selection_mode": mode,
        "repeat": TARGET_REPEAT,
        "fold": TARGET_FOLD,
        **scores,
    })

pd.DataFrame(summary_rows).to_csv(summary_output, index=False)

print("\nSaved patient comparison:")
print(patient_output)

print("\nSaved diagnostic summary:")
print(summary_output)

all_checks_passed = all(
    scores["genes_match"]
    and scores["f1_matches"]
    and scores["balanced_matches"]
    for scores in scores_by_mode.values()
)

print("\n" + "=" * 70)

if all_checks_passed:
    print("REPRODUCIBILITY CHECK PASSED")
    print("Both conditions match the original saved genes and scores.")
else:
    print("REPRODUCIBILITY CHECK FAILED")
    print("Do not interpret the patient comparison until mismatches are resolved.")

print("=" * 70)