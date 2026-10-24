from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "leukemia.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "results" / "tables"
    / "nested_cv_results.csv"
)

SEED = 42

OUTER_SPLITS = 3
INNER_SPLITS = 3

FEATURE_COUNTS = [10, 25, 50, 100]
K_VALUES = [1, 3, 5]


# =====================================================
# 2. LOAD DATA
# =====================================================

data = pd.read_csv(DATA_FILE)

X = data.drop(columns="target").to_numpy(dtype=float)

y = LabelEncoder().fit_transform(
    data["target"].astype(str).to_numpy()
)

print("=" * 65)
print("NESTED CROSS-VALIDATION EXPERIMENT")
print("=" * 65)

print(f"Patients: {X.shape[0]}")
print(f"Genes: {X.shape[1]}")
print(f"Outer folds: {OUTER_SPLITS}")
print(f"Inner folds: {INNER_SPLITS}")


# =====================================================
# 3. DEFINE THE PIPELINE
# =====================================================

def make_pipeline(selector_name):

    if selector_name == "anova":
        score_function = f_classif

    elif selector_name == "mutual_information":
        def score_function(X_train, y_train):
            return mutual_info_classif(
                X_train,
                y_train,
                random_state=SEED,
            )

    else:
        raise ValueError(
            f"Unknown selector: {selector_name}"
        )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "selector",
            SelectKBest(score_func=score_function),
        ),
        ("classifier", KNeighborsClassifier()),
    ])


# =====================================================
# 4. NESTED CROSS-VALIDATION
# =====================================================

outer_cv = StratifiedKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=SEED,
)

results = []

for selector_name in ["anova", "mutual_information"]:

    print("\n" + "=" * 65)
    print(f"SELECTOR: {selector_name.upper()}")
    print("=" * 65)

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):

        start_time = time.perf_counter()

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"\nOuter fold {fold}/{OUTER_SPLITS}")

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=SEED + fold,
        )

        pipeline = make_pipeline(selector_name)

        parameter_grid = {
            "selector__k": FEATURE_COUNTS,
            "classifier__n_neighbors": K_VALUES,
            "classifier__weights": [
                "uniform",
                "distance",
            ],
        }

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1_macro",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )

        # Tune and fit using outer training data only.
        search.fit(X_train, y_train)

        # Evaluate on held-out outer test data.
        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions,
        )

        elapsed = time.perf_counter() - start_time

        best_features = search.best_params_[
            "selector__k"
        ]

        best_neighbors = search.best_params_[
            "classifier__n_neighbors"
        ]

        best_weights = search.best_params_[
            "classifier__weights"
        ]

        results.append({
            "dataset": "Leukemia",
            "selector": selector_name,
            "outer_fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "selected_features": best_features,
            "knn_neighbors": best_neighbors,
            "knn_weights": best_weights,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "inner_best_macro_f1": search.best_score_,
            "elapsed_seconds": elapsed,
            "seed": SEED,
        })

        print(f"  Selected genes: {best_features}")
        print(f"  k-NN neighbors: {best_neighbors}")
        print(f"  Weights: {best_weights}")
        print(f"  Outer Macro-F1: {macro_f1:.4f}")
        print(
            "  Outer balanced accuracy: "
            f"{balanced_accuracy:.4f}"
        )
        print(f"  Runtime: {elapsed:.2f} seconds")


# =====================================================
# 5. SAVE RESULTS
# =====================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =====================================================
# 6. SUMMARIZE RESULTS
# =====================================================

summary = (
    results_df
    .groupby("selector")
    .agg(
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
    .reset_index()
)

print("\n" + "=" * 65)
print("NESTED CROSS-VALIDATION SUMMARY")
print("=" * 65)

print(summary.to_string(index=False))

print(f"\nResults saved to:\n{OUTPUT_FILE}")

print(
    "\nThese are preliminary nested-CV results "
    "from one seed and three outer folds."
)