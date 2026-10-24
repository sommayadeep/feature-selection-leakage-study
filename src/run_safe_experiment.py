from pathlib import Path
import time

import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    make_scorer,
    f1_score,
    balanced_accuracy_score,
)

from experiment_config import (
    DEFAULT_CONFIG,
    load_outer_split_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "leukemia.csv"
MANIFEST_FILE = PROJECT_ROOT / "configs" / "outer_fold_manifest.csv"
RESULT_FILE = (
    PROJECT_ROOT / "results" / "tables"
    / "leukemia_safe_knn_shared_3x5.csv"
)

SEED = DEFAULT_CONFIG.seed
K_FEATURES = DEFAULT_CONFIG.selected_features

# Load the validated dataset.
data = pd.read_csv(DATA_FILE)

X = data.drop(columns="target").to_numpy(dtype=float)
y_text = data["target"].to_numpy()

# Convert class names into numeric labels for model training.
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

# Each pipeline learns preprocessing and feature selection
# separately inside each training fold.
methods = {
    "all_features": "passthrough",
    "anova": SelectKBest(score_func=f_classif, k=K_FEATURES),
    "mutual_information": SelectKBest(
        score_func=lambda X_train, y_train: mutual_info_classif(
            X_train, y_train, random_state=SEED
        ),
        k=K_FEATURES,
    ),
}

outer_splits = load_outer_split_manifest(MANIFEST_FILE, y)

scoring = {
    "macro_f1": make_scorer(f1_score, average="macro"),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}

rows = []

for method_name, selector in methods.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", selector),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=DEFAULT_CONFIG.knn_neighbors,
            ),
        ),
    ])

    start = time.perf_counter()

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=outer_splits,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    elapsed = time.perf_counter() - start

    rows.append({
        "dataset": "Leukemia",
        "model": "k-NN",
        "selection": method_name,
        "n_features_selected": (
            X.shape[1] if method_name == "all_features"
            else K_FEATURES
        ),
        "macro_f1_mean": scores["test_macro_f1"].mean(),
        "macro_f1_std": scores["test_macro_f1"].std(ddof=1),
        "balanced_accuracy_mean": scores[
            "test_balanced_accuracy"
        ].mean(),
        "balanced_accuracy_std": scores[
            "test_balanced_accuracy"
        ].std(ddof=1),
        "runtime_seconds": elapsed,
        "cv_splits": DEFAULT_CONFIG.outer_splits,
        "cv_repeats": DEFAULT_CONFIG.outer_repeats,
        "seed": SEED,
    })

    print(f"\nCompleted: {method_name}")
    print(f"Macro-F1: {rows[-1]['macro_f1_mean']:.4f}")
    print(
        "Balanced accuracy: "
        f"{rows[-1]['balanced_accuracy_mean']:.4f}"
    )
    print(f"Runtime: {elapsed:.2f} seconds")

RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(rows)
results.to_csv(RESULT_FILE, index=False)

print("\n" + "=" * 60)
print("INITIAL LEAKAGE-SAFE k-NN RESULTS")
print("=" * 60)
print(results.to_string(index=False))
print(f"\nSaved results to: {RESULT_FILE}")
print(
    "\nIMPORTANT: These are preliminary pipeline-debugging results, "
    "not final conference-paper results."
)