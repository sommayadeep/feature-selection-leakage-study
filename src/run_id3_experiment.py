import argparse
from pathlib import Path
from collections import Counter
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

from experiment_config import (
    DEFAULT_CONFIG,
    load_outer_split_manifest,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leukemia.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

MANIFEST_FILE = PROJECT_ROOT / "configs" / "outer_fold_manifest.csv"

RESULTS_FILE = OUTPUT_DIR / "shared_3x5_id3_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "shared_3x5_id3_summary.csv"

FEATURE_SELECTION_FILE = (
    OUTPUT_DIR / "shared_3x5_id3_selected_features.csv"
)
PREDICTIONS_FILE = OUTPUT_DIR / "shared_3x5_id3_predictions.csv"
CONFUSION_FILE = OUTPUT_DIR / "shared_3x5_id3_confusion_matrices.csv"
SEED = DEFAULT_CONFIG.seed

N_SPLITS = DEFAULT_CONFIG.outer_splits
N_REPEATS = DEFAULT_CONFIG.outer_repeats

N_FEATURES = DEFAULT_CONFIG.selected_features
N_BINS = DEFAULT_CONFIG.discretization_bins

MAX_DEPTH = DEFAULT_CONFIG.id3_max_depth
MIN_SAMPLES_SPLIT = DEFAULT_CONFIG.id3_min_samples_split

# Both selectors will be evaluated using safe and leaky selection.
SELECTORS = ["ANOVA", "MI"]

# Each fold is evaluated under four conditions:
# ANOVA safe, ANOVA leaky, MI safe, MI leaky.


# ============================================================

# 3. ENTROPY
# ============================================================

def calculate_entropy(y):
    """
    Calculate Shannon entropy for a collection of class labels.
    """

    y = np.asarray(y)

    if len(y) == 0:
        return 0.0

    counts = Counter(y)
    total = len(y)

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return float(entropy)


# ============================================================
# 4. INFORMATION GAIN
# ============================================================

def calculate_information_gain(y, feature_values):
    """
    Calculate information gain for one categorical feature.
    """

    y = np.asarray(y)
    feature_values = np.asarray(feature_values)

    if len(y) != len(feature_values):
        raise ValueError(
            "Labels and feature values must have equal lengths."
        )

    if len(y) == 0:
        return 0.0

    parent_entropy = calculate_entropy(y)

    weighted_child_entropy = 0.0
    total_samples = len(y)

    for value in np.unique(feature_values):

        mask = feature_values == value

        child_labels = y[mask]

        proportion = len(child_labels) / total_samples

        child_entropy = calculate_entropy(child_labels)

        weighted_child_entropy += (
            proportion * child_entropy
        )

    information_gain = (
        parent_entropy - weighted_child_entropy
    )

    return max(0.0, float(information_gain))


# ============================================================
# 5. MAJORITY CLASS
# ============================================================

def majority_class(y):
    """
    Return the most frequent class.

    Counter.most_common() resolves ties according to the
    order in which classes appear.
    """

    if len(y) == 0:
        raise ValueError(
            "Cannot calculate the majority class of empty labels."
        )

    return Counter(y).most_common(1)[0][0]


# ============================================================
# 6. BUILD CUSTOM ID3-STYLE TREE
# ============================================================

def build_id3_tree(
    X,
    y,
    feature_names,
    depth=0,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
):
    """
    Build a depth-limited ID3-style decision tree.

    X must contain categorical integer features.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    feature_names = np.asarray(
        feature_names,
        dtype=object,
    )

    if len(y) == 0:
        raise ValueError(
            "Cannot build a tree from an empty dataset."
        )

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")

    if X.shape[0] != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if X.shape[1] != len(feature_names):
        raise ValueError(
            "The number of feature names must match X."
        )

    # Stop if all samples belong to one class.
    if len(np.unique(y)) == 1:
        return {"leaf": y[0]}

    # Stop if maximum depth has been reached.
    if depth >= max_depth:
        return {"leaf": majority_class(y)}

    # Stop if there are too few samples to split.
    if len(y) < min_samples_split:
        return {"leaf": majority_class(y)}

    # Stop if no features remain.
    if X.shape[1] == 0:
        return {"leaf": majority_class(y)}

    # Calculate information gain for every available feature.
    gains = np.array([
        calculate_information_gain(
            y,
            X[:, feature_index],
        )
        for feature_index in range(X.shape[1])
    ])

    best_index = int(np.argmax(gains))
    best_gain = float(gains[best_index])

    # Avoid splitting when no meaningful gain is available.
    if best_gain <= 1e-12:
        return {"leaf": majority_class(y)}

    best_feature_name = feature_names[best_index]

    tree = {
        "feature": best_feature_name,
        "majority": majority_class(y),
        "gain": best_gain,
        "branches": {},
    }

    # Remove the selected feature before growing child nodes.
    remaining_features = np.delete(
        feature_names,
        best_index,
    )

    remaining_X = np.delete(
        X,
        best_index,
        axis=1,
    )

    # Build one branch for every observed feature category.
    for value in np.unique(X[:, best_index]):

        mask = X[:, best_index] == value

        child_X = remaining_X[mask]
        child_y = y[mask]

        child_tree = build_id3_tree(
            child_X,
            child_y,
            remaining_features,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        # Convert NumPy scalar values to ordinary Python scalars.
        branch_value = (
            value.item()
            if hasattr(value, "item")
            else value
        )

        tree["branches"][branch_value] = child_tree

    return tree


# ============================================================
# 7. PREDICTION
# ============================================================

def predict_one(tree, sample, feature_names):
    """
    Predict the class of one sample.
    """

    if "leaf" in tree:
        return tree["leaf"]

    feature_name = tree["feature"]

    feature_names = list(feature_names)

    feature_index = feature_names.index(feature_name)

    feature_value = sample[feature_index]

    if feature_value in tree["branches"]:

        child_tree = tree["branches"][feature_value]

        remaining_sample = np.delete(
            np.asarray(sample),
            feature_index,
        )

        remaining_features = np.delete(
            np.asarray(feature_names, dtype=object),
            feature_index,
        )

        return predict_one(
            child_tree,
            remaining_sample,
            remaining_features,
        )

    # Handle a category that was not observed during training.
    return tree["majority"]


def predict_batch(tree, X, feature_names):
    """
    Predict the classes of multiple samples.
    """

    return np.asarray([
        predict_one(
            tree,
            sample,
            feature_names,
        )
        for sample in X
    ])


# ============================================================
# 8. SELECTOR FACTORY
# ============================================================

def create_selector(selector_name):
    """
    Create an ANOVA or mutual-information selector.
    """

    if selector_name == "ANOVA":

        return SelectKBest(
            score_func=f_classif,
            k=N_FEATURES,
        )

    if selector_name == "MI":

        return SelectKBest(
            score_func=lambda features, labels: (
                mutual_info_classif(
                    features,
                    labels,
                    random_state=SEED,
                )
            ),
            k=N_FEATURES,
        )

    raise ValueError(
        f"Unknown selector: {selector_name}"
    )


def select_fold_features(
    X_train,
    X_test,
    y_train,
    selector_name,
    gene_names,
    *,
    leakage_mode="safe",
    X_all=None,
    y_all=None,
):
    """Select fold features, keeping held-out labels out of the safe path."""

    selector = create_selector(selector_name)
    if leakage_mode == "safe":
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif leakage_mode == "leaky":
        if X_all is None or y_all is None:
            raise ValueError(
                "The leaky diagnostic requires all observations and labels."
            )
        selector.fit(X_all, y_all)
        selected_indices = selector.get_support(indices=True)
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
    else:
        raise ValueError(f"Unknown leakage mode: {leakage_mode}")

    selected_indices = selector.get_support(indices=True)
    return (
        X_train_selected,
        X_test_selected,
        np.asarray(gene_names)[selected_indices],
        selected_indices,
    )


# ============================================================


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the shared ID3 leakage comparison.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N manifest folds for a smoke test.",
    )
    parser.add_argument(
        "--output-stem",
        default="shared_3x5_id3",
        help="Stem for generated result files.",
    )
    args = parser.parse_args(argv)

    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("--max-folds must be positive.")

    results_file = OUTPUT_DIR / f"{args.output_stem}_results.csv"
    summary_file = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    feature_selection_file = (
        OUTPUT_DIR / f"{args.output_stem}_selected_features.csv"
    )
    predictions_file = OUTPUT_DIR / f"{args.output_stem}_predictions.csv"
    confusion_file = (
        OUTPUT_DIR / f"{args.output_stem}_confusion_matrices.csv"
    )

    # 2. LOAD DATA
    # ============================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check the location of leukemia.csv."
        )

    data = pd.read_csv(DATA_FILE)

    if "target" not in data.columns:
        raise ValueError(
            "The dataset must contain a column named 'target'."
        )

    if data["target"].isna().any():
        raise ValueError(
            "The target column contains missing labels."
        )

    X = data.drop(columns="target").to_numpy(dtype=float)

    y = LabelEncoder().fit_transform(
        data["target"].astype(str).to_numpy()
    )

    gene_names = (
        data.drop(columns="target")
        .columns
        .to_numpy()
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "The dataset must contain at least two classes."
        )

    if not 1 <= N_FEATURES <= X.shape[1]:
        raise ValueError(
            "N_FEATURES must be between 1 and the total "
            "number of genes."
        )

    if N_SPLITS < 2:
        raise ValueError("N_SPLITS must be at least 2.")

    if np.bincount(y).min() < N_SPLITS:
        raise ValueError(
            "Each class must have at least N_SPLITS patients."
        )

    print("=" * 70)
    print("ID3-STYLE TREE: FEATURE-SELECTION LEAKAGE EXPERIMENT")
    print("=" * 70)

    print(f"Dataset: {DATA_FILE.name}")
    print(f"Patients: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Selected features: {N_FEATURES}")
    print(f"Discretization bins: {N_BINS}")
    print(f"Maximum tree depth: {MAX_DEPTH}")
    print(f"Cross-validation: {N_SPLITS} folds x {N_REPEATS} repeats")


    # ============================================================
    # 9. TEST ENTROPY, INFORMATION GAIN AND TREE
    # ============================================================

    print("\n" + "=" * 70)
    print("TESTING THE ID3 IMPLEMENTATION")
    print("=" * 70)

    balanced_entropy = calculate_entropy(
        ["ALL", "ALL", "AML", "AML"]
    )

    pure_entropy = calculate_entropy(
        ["ALL", "ALL", "ALL", "ALL"]
    )

    perfect_gain = calculate_information_gain(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
    )

    print(f"Balanced-class entropy: {balanced_entropy:.4f}")
    print(f"Pure-class entropy: {pure_entropy:.4f}")
    print(f"Perfect-split information gain: {perfect_gain:.4f}")

    X_toy = np.array([
        [0],
        [0],
        [1],
        [1],
    ])

    y_toy = np.array([0, 0, 1, 1])

    toy_features = np.array(["toy_gene"])

    toy_tree = build_id3_tree(
        X_toy,
        y_toy,
        toy_features,
    )

    toy_predictions = predict_batch(
        toy_tree,
        X_toy,
        toy_features,
    )

    print("Toy predictions:", toy_predictions.tolist())
    print("Actual labels:", y_toy.tolist())

    if not np.array_equal(toy_predictions, y_toy):
        raise RuntimeError(
            "The toy test failed. Check the ID3 implementation."
        )

    print("Toy test passed.")


    # ============================================================
    # 10. SHARED REPEATED STRATIFIED OUTER SPLITS
    # ============================================================

    fold_splits = load_outer_split_manifest(
        MANIFEST_FILE,
        y,
    )
    if args.max_folds is not None:
        fold_splits = fold_splits[:args.max_folds]

    results = []
    feature_selection_records = []
    prediction_records = []
    confusion_records = []

    print("\n" + "=" * 70)
    print("RUNNING THE EXPERIMENT")
    print("=" * 70)

    for fold_number, (train_idx, test_idx) in enumerate(
        fold_splits,
        start=1,
    ):

        repeat_number = (
            (fold_number - 1) // N_SPLITS
        ) + 1

        fold_in_repeat = (
            (fold_number - 1) % N_SPLITS
        ) + 1

        print(
            f"\nRepeat {repeat_number}/{N_REPEATS}, "
            f"Fold {fold_in_repeat}/{N_SPLITS}"
        )

        # --------------------------------------------------------
        # A. Split the patients.
        # --------------------------------------------------------

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # --------------------------------------------------------
        # B. Fit imputation on training patients only.
        # --------------------------------------------------------

        imputer = SimpleImputer(
            strategy="median",
        )

        X_train_imputed = imputer.fit_transform(
            X_train_raw
        )

        X_test_imputed = imputer.transform(
            X_test_raw
        )

        # Transform all patients using the imputer fitted on
        # the current training fold. This does NOT fit the
        # imputer on the test patients.
        X_all_imputed = imputer.transform(X)

        # --------------------------------------------------------
        # C. Run ANOVA and MI under safe and leaky selection.
        # --------------------------------------------------------

        for selector_name in SELECTORS:

            for leakage_mode in ["safe", "leaky"]:

                print(
                    f"  Selector={selector_name}, "
                    f"mode={leakage_mode}"
                )

                # -----------------------------------------------
                # SAFE FEATURE SELECTION
                # -----------------------------------------------
                # The leaky diagnostic receives all labels explicitly;
                # the safe branch receives training-fold labels only.
                selection_kwargs = (
                    {
                        "X_all": X_all_imputed,
                        "y_all": y,
                    }
                    if leakage_mode == "leaky"
                    else {}
                )
                (
                    X_train_selected,
                    X_test_selected,
                    selected_gene_names,
                    selected_indices,
                ) = select_fold_features(
                    X_train_imputed,
                    X_test_imputed,
                    y_train,
                    selector_name,
                    gene_names,
                    leakage_mode=leakage_mode,
                    **selection_kwargs,
                )

                # -----------------------------------------------
                # D. Fit discretization on training data only.
                # -----------------------------------------------

                discretizer = KBinsDiscretizer(
                    n_bins=N_BINS,
                    encode="ordinal",
                    strategy="quantile",
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                    )

                    X_train_discrete = (
                        discretizer.fit_transform(
                            X_train_selected
                        )
                    )

                    X_test_discrete = (
                        discretizer.transform(
                            X_test_selected
                        )
                    )

                X_train_discrete = (
                    X_train_discrete.astype(int)
                )

                X_test_discrete = (
                    X_test_discrete.astype(int)
                )

                # -----------------------------------------------
                # E. Build the ID3-style tree.
                # -----------------------------------------------

                tree = build_id3_tree(
                    X_train_discrete,
                    y_train,
                    selected_gene_names,
                    max_depth=MAX_DEPTH,
                    min_samples_split=MIN_SAMPLES_SPLIT,
                )

                # -----------------------------------------------
                # F. Predict held-out patients.
                # -----------------------------------------------

                predictions = predict_batch(
                    tree,
                    X_test_discrete,
                    selected_gene_names,
                )

                confusion_values = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1],
                ).ravel()

                # -----------------------------------------------
                # G. Evaluate.
                # -----------------------------------------------

                macro_f1 = f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )

                balanced_accuracy = (
                    balanced_accuracy_score(
                        y_test,
                        predictions,
                    )
                )

                # -----------------------------------------------
                # H. Store results.
                # -----------------------------------------------

                results.append({
                    "dataset": "Leukemia",
                    "classifier": "Custom ID3-style",
                    "selector": selector_name,
                    "selection_mode": leakage_mode,
                    "repeat": repeat_number,
                    "fold": fold_in_repeat,
                    "global_fold": fold_number,
                    "features_selected": len(
                        selected_gene_names
                    ),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "macro_f1": macro_f1,
                    "balanced_accuracy": balanced_accuracy,
                })

                feature_selection_records.append({
                    "dataset": "Leukemia",
                    "selector": selector_name,
                    "selection_mode": leakage_mode,
                    "repeat": repeat_number,
                    "fold": fold_in_repeat,
                    "global_fold": fold_number,
                    "selected_genes": ";".join(
                        selected_gene_names.astype(str)
                    ),
                })

                for patient_index, true_label, predicted_label in zip(
                    test_idx,
                    y_test,
                    predictions,
                ):
                    prediction_records.append({
                        "dataset": "Leukemia",
                        "selector": selector_name,
                        "selection_mode": leakage_mode,
                        "repeat": repeat_number,
                        "fold": fold_in_repeat,
                        "global_fold": fold_number,
                        "patient_index": int(patient_index),
                        "true_label": int(true_label),
                        "predicted_label": int(predicted_label),
                    })

                confusion_records.append({
                    "dataset": "Leukemia",
                    "selector": selector_name,
                    "selection_mode": leakage_mode,
                    "repeat": repeat_number,
                    "fold": fold_in_repeat,
                    "global_fold": fold_number,
                    "true_negative": int(confusion_values[0]),
                    "false_positive": int(confusion_values[1]),
                    "false_negative": int(confusion_values[2]),
                    "true_positive": int(confusion_values[3]),
                })

                print(
                    f"    Macro-F1: {macro_f1:.4f} | "
                    f"Balanced accuracy: "
                    f"{balanced_accuracy:.4f}"
                )


    # ============================================================
    # 11. SAVE FOLD-LEVEL RESULTS
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        results_file,
        index=False,
    )
    feature_selection_df = pd.DataFrame(
        feature_selection_records
    )

    feature_selection_df.to_csv(
        feature_selection_file,
        index=False,
    )

    pd.DataFrame(prediction_records).to_csv(
        predictions_file,
        index=False,
    )
    pd.DataFrame(confusion_records).to_csv(
        confusion_file,
        index=False,
    )

    print(
        f"Feature selection records saved to: "
        f"{feature_selection_file}"
    )


    # ============================================================
    # 12. SUMMARIZE RESULTS
    # ============================================================

    summary_df = (
        results_df
        .groupby(
            ["selector", "selection_mode"],
            as_index=False,
        )
        .agg(
            folds=("macro_f1", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
        )
    )

    # These differences are descriptive comparisons.
    # They are NOT statistical significance tests.

    inflation_rows = []

    for selector_name in SELECTORS:

        selector_results = summary_df[
            summary_df["selector"] == selector_name
        ]

        safe_rows = selector_results[
            selector_results["selection_mode"] == "safe"
        ]

        leaky_rows = selector_results[
            selector_results["selection_mode"] == "leaky"
        ]

        if len(safe_rows) == 1 and len(leaky_rows) == 1:

            safe_row = safe_rows.iloc[0]
            leaky_row = leaky_rows.iloc[0]

            inflation_rows.append({
                "selector": selector_name,
                "macro_f1_safe": safe_row["macro_f1_mean"],
                "macro_f1_leaky": leaky_row["macro_f1_mean"],
                "macro_f1_difference": (
                    leaky_row["macro_f1_mean"]
                    - safe_row["macro_f1_mean"]
                ),
                "balanced_accuracy_safe": (
                    safe_row["balanced_accuracy_mean"]
                ),
                "balanced_accuracy_leaky": (
                    leaky_row["balanced_accuracy_mean"]
                ),
                "balanced_accuracy_difference": (
                    leaky_row["balanced_accuracy_mean"]
                    - safe_row["balanced_accuracy_mean"]
                ),
            })

    inflation_df = pd.DataFrame(
        inflation_rows
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )


    # ============================================================
    # 13. PRINT FINAL RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print("ID3 EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("DESCRIPTIVE PERFORMANCE-INFLATION COMPARISON")
    print("=" * 70)

    if not inflation_df.empty:

        print(
            inflation_df.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        print(
            "\nA positive difference means the leaky procedure "
            "reported a higher mean score."
        )

        print(
            "These differences are descriptive and do not "
            "establish statistical significance."
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"Fold-level results:\n{results_file}")
    print(f"\nSummary results:\n{summary_file}")
    print(f"\nPredictions:\n{predictions_file}")
    print(f"\nConfusion matrices:\n{confusion_file}")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
