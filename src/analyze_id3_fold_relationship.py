from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLES_DIR = PROJECT_ROOT / "results" / "tables"

PERFORMANCE_FILE = TABLES_DIR / "id3_experiment_results.csv"
OVERLAP_FILE = TABLES_DIR / "id3_feature_overlap.csv"

OUTPUT_FILE = TABLES_DIR / "id3_fold_relationship.csv"


# ============================================================
# 2. LOAD RESULTS
# ============================================================

performance = pd.read_csv(PERFORMANCE_FILE)
overlap = pd.read_csv(OVERLAP_FILE)

print("=" * 70)
print("FOLD-MATCHED PERFORMANCE AND FEATURE-OVERLAP ANALYSIS")
print("=" * 70)

print(f"Performance rows: {len(performance)}")
print(f"Overlap rows: {len(overlap)}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

performance_required = {
    "selector",
    "selection_mode",
    "repeat",
    "fold",
    "macro_f1",
    "balanced_accuracy",
}

overlap_required = {
    "selector",
    "repeat",
    "fold",
    "jaccard_similarity",
    "common_gene_count",
}

if not performance_required.issubset(performance.columns):
    raise ValueError(
        "Missing performance columns: "
        f"{performance_required - set(performance.columns)}"
    )

if not overlap_required.issubset(overlap.columns):
    raise ValueError(
        "Missing overlap columns: "
        f"{overlap_required - set(overlap.columns)}"
    )


# ============================================================
# 4. MATCH SAFE AND LEAKY RESULTS BY FOLD
# ============================================================

fold_keys = ["selector", "repeat", "fold"]

safe = performance[
    performance["selection_mode"] == "safe"
].copy()

leaky = performance[
    performance["selection_mode"] == "leaky"
].copy()

if safe.duplicated(fold_keys).any():
    raise ValueError("Duplicate safe performance records found.")

if leaky.duplicated(fold_keys).any():
    raise ValueError("Duplicate leaky performance records found.")

safe = safe[
    fold_keys + ["macro_f1", "balanced_accuracy"]
].rename(
    columns={
        "macro_f1": "safe_macro_f1",
        "balanced_accuracy": "safe_balanced_accuracy",
    }
)

leaky = leaky[
    fold_keys + ["macro_f1", "balanced_accuracy"]
].rename(
    columns={
        "macro_f1": "leaky_macro_f1",
        "balanced_accuracy": "leaky_balanced_accuracy",
    }
)

matched = safe.merge(
    leaky,
    on=fold_keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

if not (matched["_merge"] == "both").all():
    raise ValueError(
        "Some folds do not have matching safe and leaky results."
    )

matched = matched.drop(columns="_merge")

matched["macro_f1_difference"] = (
    matched["leaky_macro_f1"] - matched["safe_macro_f1"]
)

matched["balanced_accuracy_difference"] = (
    matched["leaky_balanced_accuracy"]
    - matched["safe_balanced_accuracy"]
)


# ============================================================
# 5. MATCH FEATURE OVERLAP TO THE SAME FOLD
# ============================================================

if overlap.duplicated(fold_keys).any():
    raise ValueError("Duplicate feature-overlap records found.")

overlap_columns = fold_keys + [
    "jaccard_similarity",
    "common_gene_count",
]

matched = matched.merge(
    overlap[overlap_columns],
    on=fold_keys,
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not (matched["_merge"] == "both").all():
    raise ValueError(
        "Some performance folds have no matching overlap record."
    )

matched = matched.drop(columns="_merge")

matched = matched.sort_values(
    ["selector", "repeat", "fold"]
).reset_index(drop=True)


# ============================================================
# 6. SAVE FOLD-LEVEL DATA
# ============================================================

matched.to_csv(OUTPUT_FILE, index=False)

print(f"\nMatched fold records: {len(matched)}")
print(f"Saved fold-level results to:\n{OUTPUT_FILE}")


# ============================================================
# 7. SUMMARIZE RESULTS BY SELECTOR
# ============================================================

summary = (
    matched.groupby("selector")
    .agg(
        folds=("macro_f1_difference", "count"),
        mean_macro_f1_difference=("macro_f1_difference", "mean"),
        median_macro_f1_difference=("macro_f1_difference", "median"),
        mean_balanced_accuracy_difference=(
            "balanced_accuracy_difference", "mean"
        ),
        mean_jaccard=("jaccard_similarity", "mean"),
        mean_common_genes=("common_gene_count", "mean"),
    )
    .reset_index()
)

print("\nSUMMARY BY SELECTOR")
print("-" * 70)
print(summary.round(4).to_string(index=False))


# ============================================================
# 8. EXPLORE THE RELATIONSHIP
# ============================================================

print("\nDESCRIPTIVE CORRELATIONS")
print("-" * 70)

for selector, group in matched.groupby("selector"):

    if (
        group["macro_f1_difference"].nunique() > 1
        and group["jaccard_similarity"].nunique() > 1
    ):
        correlation = group[
            ["macro_f1_difference", "jaccard_similarity"]
        ].corr(method="spearman").iloc[0, 1]

        print(
            f"{selector}: Spearman correlation = {correlation:.4f}"
        )
    else:
        print(
            f"{selector}: Correlation unavailable because "
            "one variable has no variation."
        )


# ============================================================
# 9. VALIDATION
# ============================================================

print("\nVALIDATION")
print("-" * 70)

print("Expected 30 matched folds:", len(matched) == 30)

print(
    "All Jaccard values valid:",
    matched["jaccard_similarity"].between(0, 1).all(),
)

print(
    "No missing matched values:",
    not matched[
        [
            "safe_macro_f1",
            "leaky_macro_f1",
            "jaccard_similarity",
        ]
    ].isna().any().any(),
)

print("\nAnalysis completed.")
print(
    "Note: Correlations are exploratory and descriptive. "
    "Repeated CV folds are dependent, so these values must not "
    "be interpreted as independent-sample significance tests."
)