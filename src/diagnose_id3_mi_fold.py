from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# Import the exact implementation used by the original experiment.
from run_id3_experiment import (
    build_id3_tree,
    predict_batch,
    create_selector,
    SEED,
    N_SPLITS,
    N_REPEATS,
    N_FEATURES,
    N_BINS,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
)

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

DATA_FILE = ROOT / "data" / "processed" / "leukemia.csv"
RESULTS_FILE = TABLES / "id3_experiment_results.csv"
GENES_FILE = TABLES / "id3_feature_selection_records.csv"

TARGET_REPEAT = 3
TARGET_FOLD = 2

data = pd.read_csv(DATA_FILE)

X = data.drop(columns="target").to_numpy(dtype=float)
gene_names = data.drop(columns="target").columns.to_numpy()
y = LabelEncoder().fit_transform(data["target"].astype(str))

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

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)
X_all_imputed = imputer.transform(X)

print("=" * 70)
print("REPRODUCIBILITY CHECK: ORIGINAL ID3 IMPLEMENTATION")
print("=" * 70)
print(f"Repeat: {TARGET_REPEAT}, fold: {TARGET_FOLD}")
print(f"Global fold: {global_fold}")
print(f"Training patients: {len(train_idx)}")
print(f"Test patients: {len(test_idx)}")

all_passed = True

for mode in ["safe", "leaky"]:
    selector = create_selector("MI")

    if mode == "safe":
        X_train_selected = selector.fit_transform(
            X_train_imputed, y_train
        )
        X_test_selected = selector.transform(X_test_imputed)
    else:
        # Reproduce the original experiment's deliberately leaky
        # feature selection: fit on all patients and labels.
        selector.fit(X_all_imputed, y)
        selected_indices = selector.get_support(indices=True)
        X_train_selected = X_train_imputed[:, selected_indices]
        X_test_selected = X_test_imputed[:, selected_indices]

    selected_indices = selector.get_support(indices=True)
    selected_gene_names = gene_names[selected_indices]

    saved_row = saved_genes[
        (saved_genes["selector"] == "MI")
        & (saved_genes["selection_mode"] == mode)
        & (saved_genes["repeat"] == TARGET_REPEAT)
        & (saved_genes["fold"] == TARGET_FOLD)
    ]

    if len(saved_row) != 1:
        raise ValueError(
            f"Expected exactly one saved gene record for {mode}; "
            f"found {len(saved_row)}."
        )

    saved_gene_list = saved_row.iloc[0]["selected_genes"].split(";")

    same_genes = set(selected_gene_names) == set(saved_gene_list)

    print(f"\n--- {mode.upper()} ---")
    print(f"Selected genes: {len(selected_gene_names)}")
    print(f"Exact gene set matches original: {same_genes}")

    if not same_genes:
        all_passed = False
        print("Genes missing from reproduction:",
              sorted(set(saved_gene_list) - set(selected_gene_names)))
        print("Unexpected reproduced genes:",
              sorted(set(selected_gene_names) - set(saved_gene_list)))

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
        selected_gene_names,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    predictions = predict_batch(
        tree,
        X_test_discrete,
        selected_gene_names,
    )

    reproduced_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    reproduced_balanced = balanced_accuracy_score(y_test, predictions)

    result_row = saved_results[
        (saved_results["selector"] == "MI")
        & (saved_results["selection_mode"] == mode)
        & (saved_results["repeat"] == TARGET_REPEAT)
        & (saved_results["fold"] == TARGET_FOLD)
    ]

    if len(result_row) != 1:
        raise ValueError(
            f"Expected exactly one saved score for {mode}; "
            f"found {len(result_row)}."
        )

    original_f1 = float(result_row.iloc[0]["macro_f1"])
    original_balanced = float(result_row.iloc[0]["balanced_accuracy"])

    f1_matches = np.isclose(reproduced_f1, original_f1)
    balanced_matches = np.isclose(reproduced_balanced, original_balanced)

    print(f"Original Macro-F1:      {original_f1:.4f}")
    print(f"Reproduced Macro-F1:    {reproduced_f1:.4f}")
    print(f"Macro-F1 matches:       {f1_matches}")
    print(f"Original balanced acc.: {original_balanced:.4f}")
    print(f"Reproduced balanced acc.: {reproduced_balanced:.4f}")
    print(f"Balanced accuracy matches: {balanced_matches}")

    if not (f1_matches and balanced_matches):
        all_passed = False

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=LabelEncoder().fit(data["target"]).classes_,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

if all_passed:
    print("All checks passed. The diagnostic reproduces the saved results.")
else:
    print(
        "Some checks failed. Do not interpret patient-level predictions "
        "yet; inspect the differences before proceeding."
    )