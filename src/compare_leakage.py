"""Run the shared fixed-configuration k-NN leakage comparison."""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiment_config import (
    DEFAULT_CONFIG,
    load_outer_split_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "leukemia.csv"
MANIFEST_FILE = PROJECT_ROOT / "configs" / "outer_fold_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

SEED = DEFAULT_CONFIG.seed
K_FEATURES = DEFAULT_CONFIG.selected_features
SELECTORS = ["anova", "mutual_information"]


def make_selector(method):
    if method == "anova":
        return SelectKBest(score_func=f_classif, k=K_FEATURES)

    if method == "mutual_information":
        return SelectKBest(
            score_func=lambda X_train, y_train: mutual_info_classif(
                X_train,
                y_train,
                random_state=SEED,
            ),
            k=K_FEATURES,
        )

    raise ValueError(f"Unknown selector: {method}")


def make_knn_classifier():
    """Create the classifier shared by both selection conditions."""

    return KNeighborsClassifier(
        n_neighbors=DEFAULT_CONFIG.knn_neighbors,
    )


def preprocess_fold(
    X_train,
    X_test,
    X_all,
):
    """Fit preprocessing on training data and transform all fold views."""

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_all_imputed = imputer.transform(X_all)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_all_scaled = scaler.transform(X_all_imputed)

    return (
        X_train_scaled,
        X_test_scaled,
        X_all_scaled,
        imputer,
        scaler,
    )


def fit_knn_condition(
    X_train_scaled,
    X_test_scaled,
    y_train,
    selection_features,
    selection_labels,
    method,
):
    """Select features, then fit k-NN using training observations only."""

    selector = make_selector(method)
    selector.fit(selection_features, selection_labels)
    selected_indices = selector.get_support(indices=True)

    classifier = make_knn_classifier()
    classifier.fit(X_train_scaled[:, selected_indices], y_train)
    predictions = classifier.predict(
        X_test_scaled[:, selected_indices],
    )
    return predictions, selected_indices


def run_experiment(
    max_folds=None,
    output_stem="shared_3x5_knn",
):
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive.")

    data = pd.read_csv(DATA_FILE)
    if "target" not in data.columns:
        raise ValueError("The dataset must contain a target column.")

    X = data.drop(columns="target").to_numpy(dtype=float)
    gene_names = data.drop(columns="target").columns.to_numpy()
    y = LabelEncoder().fit_transform(data["target"].astype(str).to_numpy())

    splits = load_outer_split_manifest(MANIFEST_FILE, y)
    if max_folds is not None:
        splits = splits[:max_folds]

    results = []
    selected_feature_records = []
    prediction_records = []
    confusion_records = []

    for method in SELECTORS:
        for fold_number, (train_idx, test_idx) in enumerate(
            splits,
            start=1,
        ):
            repeat_number = (
                (fold_number - 1) // DEFAULT_CONFIG.outer_splits
            ) + 1
            fold_in_repeat = (
                (fold_number - 1) % DEFAULT_CONFIG.outer_splits
            ) + 1
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            (
                X_train_scaled,
                X_test_scaled,
                X_all_scaled,
                imputer,
                scaler,
            ) = preprocess_fold(
                X_train,
                X_test,
                X,
            )

            safe_start = time.perf_counter()
            safe_predictions, safe_indices = fit_knn_condition(
                X_train_scaled,
                X_test_scaled,
                y_train,
                X_train_scaled,
                y_train,
                method,
            )
            safe_elapsed = time.perf_counter() - safe_start

            leaky_start = time.perf_counter()
            leaky_predictions, leaky_indices = fit_knn_condition(
                X_train_scaled,
                X_test_scaled,
                y_train,
                X_all_scaled,
                y,
                method,
            )
            leaky_elapsed = time.perf_counter() - leaky_start

            for mode, predictions, indices, elapsed in (
                (
                    "safe",
                    safe_predictions,
                    safe_indices,
                    safe_elapsed,
                ),
                (
                    "leaky",
                    leaky_predictions,
                    leaky_indices,
                    leaky_elapsed,
                ),
            ):
                confusion_values = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1],
                ).ravel()
                results.append({
                    "dataset": "Leukemia",
                    "model": "k-NN",
                    "selector": method,
                    "selection_mode": mode,
                    "repeat": repeat_number,
                    "fold": fold_in_repeat,
                    "global_fold": fold_number,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "features_selected": len(indices),
                    "macro_f1": f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_test,
                        predictions,
                    ),
                    "fit_predict_seconds": elapsed,
                })
                selected_feature_records.append({
                    "dataset": "Leukemia",
                    "model": "k-NN",
                    "selector": method,
                    "selection_mode": mode,
                    "repeat": repeat_number,
                    "fold": fold_in_repeat,
                    "global_fold": fold_number,
                    "selected_genes": ";".join(
                        gene_names[indices].astype(str)
                    ),
                })
                for patient_index, true_label, predicted_label in zip(
                    test_idx,
                    y_test,
                    predictions,
                ):
                    prediction_records.append({
                        "dataset": "Leukemia",
                        "model": "k-NN",
                        "selector": method,
                        "selection_mode": mode,
                        "repeat": repeat_number,
                        "fold": fold_in_repeat,
                        "global_fold": fold_number,
                        "patient_index": int(patient_index),
                        "true_label": int(true_label),
                        "predicted_label": int(predicted_label),
                    })
                confusion_records.append({
                    "dataset": "Leukemia",
                    "model": "k-NN",
                    "selector": method,
                    "selection_mode": mode,
                    "repeat": repeat_number,
                    "fold": fold_in_repeat,
                    "global_fold": fold_number,
                    "true_negative": int(confusion_values[0]),
                    "false_positive": int(confusion_values[1]),
                    "false_negative": int(confusion_values[2]),
                    "true_positive": int(confusion_values[3]),
                })

            print(
                f"{method} fold {fold_number}/{len(splits)} complete"
            )

    output_paths = {
        "results": OUTPUT_DIR / f"{output_stem}_results.csv",
        "summary": OUTPUT_DIR / f"{output_stem}_summary.csv",
        "selected": OUTPUT_DIR / f"{output_stem}_selected_features.csv",
        "predictions": OUTPUT_DIR / f"{output_stem}_predictions.csv",
        "confusion": OUTPUT_DIR / f"{output_stem}_confusion_matrices.csv",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_paths["results"], index=False)
    pd.DataFrame(selected_feature_records).to_csv(
        output_paths["selected"],
        index=False,
    )
    pd.DataFrame(prediction_records).to_csv(
        output_paths["predictions"],
        index=False,
    )
    pd.DataFrame(confusion_records).to_csv(
        output_paths["confusion"],
        index=False,
    )

    summary_df = (
        results_df
        .groupby(["selector", "selection_mode"], as_index=False)
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
    summary_df.to_csv(output_paths["summary"], index=False)

    print("\nSaved shared k-NN files:")
    for path in output_paths.values():
        print(path)
    return output_paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the shared k-NN leakage comparison.",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--output-stem",
        default="shared_3x5_knn",
    )
    args = parser.parse_args(argv)
    run_experiment(
        max_folds=args.max_folds,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
