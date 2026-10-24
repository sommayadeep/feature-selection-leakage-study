from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLES_DIR = PROJECT_ROOT / "results" / "tables"

PERFORMANCE_FILE = TABLES_DIR / "id3_experiment_results.csv"

OVERLAP_FILE = TABLES_DIR / "id3_feature_overlap.csv"

OUTPUT_FILE = TABLES_DIR / "id3_combined_analysis.csv"


# ============================================================
# 2. LOAD EXISTING RESULTS
# ============================================================

print("=" * 70)
print("ID3: PERFORMANCE AND FEATURE-SELECTION OVERLAP ANALYSIS")
print("=" * 70)

if not PERFORMANCE_FILE.exists():
    raise FileNotFoundError(
        f"Performance results not found: {PERFORMANCE_FILE}"
    )

if not OVERLAP_FILE.exists():
    raise FileNotFoundError(
        f"Feature-overlap results not found: {OVERLAP_FILE}"
    )

performance_df = pd.read_csv(PERFORMANCE_FILE)
overlap_df = pd.read_csv(OVERLAP_FILE)

print(f"\nPerformance records loaded: {len(performance_df)}")
print(f"Feature-overlap records loaded: {len(overlap_df)}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_performance_columns = {
    "selector",
    "selection_mode",
    "macro_f1",
    "balanced_accuracy",
}

required_overlap_columns = {
    "selector",
    "jaccard_similarity",
    "common_gene_count",
}

missing_performance = (
    required_performance_columns - set(performance_df.columns)
)

missing_overlap = (
    required_overlap_columns - set(overlap_df.columns)
)

if missing_performance:
    raise ValueError(
        f"Missing columns in performance file: "
        f"{sorted(missing_performance)}"
    )

if missing_overlap:
    raise ValueError(
        f"Missing columns in overlap file: "
        f"{sorted(missing_overlap)}"
    )

if not set(performance_df["selection_mode"]) >= {"safe", "leaky"}:
    raise ValueError(
        "Performance results must contain both safe and leaky modes."
    )


# ============================================================
# 4. CALCULATE PERFORMANCE SUMMARY
# ============================================================

performance_summary = (
    performance_df
    .groupby(["selector", "selection_mode"])
    .agg(
        folds=("macro_f1", "count"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=(
            "balanced_accuracy", "mean"
        ),
        balanced_accuracy_std=(
            "balanced_accuracy", "std"
        ),
    )
    .reset_index()
)

# Separate safe and leaky results.

safe_results = performance_summary[
    performance_summary["selection_mode"] == "safe"
].drop(columns="selection_mode")

safe_results = safe_results.rename(
    columns={
        "folds": "safe_folds",
        "macro_f1_mean": "safe_macro_f1",
        "macro_f1_std": "safe_macro_f1_std",
        "balanced_accuracy_mean": "safe_balanced_accuracy",
        "balanced_accuracy_std": "safe_balanced_accuracy_std",
    }
)

leaky_results = performance_summary[
    performance_summary["selection_mode"] == "leaky"
].drop(columns="selection_mode")

leaky_results = leaky_results.rename(
    columns={
        "folds": "leaky_folds",
        "macro_f1_mean": "leaky_macro_f1",
        "macro_f1_std": "leaky_macro_f1_std",
        "balanced_accuracy_mean": "leaky_balanced_accuracy",
        "balanced_accuracy_std": "leaky_balanced_accuracy_std",
    }
)


# ============================================================
# 5. COMBINE SAFE AND LEAKY PERFORMANCE
# ============================================================

combined = safe_results.merge(
    leaky_results,
    on="selector",
    how="inner",
    validate="one_to_one",
)

# Positive differences indicate higher scores under leaky selection.

combined["macro_f1_difference"] = (
    combined["leaky_macro_f1"]
    - combined["safe_macro_f1"]
)

combined["balanced_accuracy_difference"] = (
    combined["leaky_balanced_accuracy"]
    - combined["safe_balanced_accuracy"]
)


# ============================================================
# 6. SUMMARIZE FEATURE-SELECTION OVERLAP
# ============================================================

overlap_summary = (
    overlap_df
    .groupby("selector")
    .agg(
        common_genes_mean=("common_gene_count", "mean"),
        jaccard_mean=("jaccard_similarity", "mean"),
        jaccard_std=("jaccard_similarity", "std"),
    )
    .reset_index()
)

combined = combined.merge(
    overlap_summary,
    on="selector",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 7. ADD DESCRIPTIVE INTERPRETATION
# ============================================================

combined["macro_f1_change_percentage_points"] = (
    combined["macro_f1_difference"] * 100
)

combined["balanced_accuracy_change_percentage_points"] = (
    combined["balanced_accuracy_difference"] * 100
)

combined["feature_overlap_description"] = combined[
    "jaccard_mean"
].apply(
    lambda value: (
        "Higher overlap"
        if value >= 0.60
        else "Moderate overlap"
        if value >= 0.40
        else "Lower overlap"
    )
)


# ============================================================
# 8. SAVE COMBINED RESULTS
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("COMBINED RESULTS")
print("=" * 70)

display_columns = [
    "selector",
    "safe_macro_f1",
    "leaky_macro_f1",
    "macro_f1_difference",
    "safe_balanced_accuracy",
    "leaky_balanced_accuracy",
    "balanced_accuracy_difference",
    "common_genes_mean",
    "jaccard_mean",
]

print(
    combined[display_columns]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 9. DISPLAY INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

for _, row in combined.iterrows():

    print(f"\nSelector: {row['selector']}")

    print(
        f"  Safe Macro-F1: "
        f"{row['safe_macro_f1']:.4f}"
    )

    print(
        f"  Leaky Macro-F1: "
        f"{row['leaky_macro_f1']:.4f}"
    )

    print(
        f"  Macro-F1 difference: "
        f"{row['macro_f1_difference']:+.4f}"
    )

    print(
        f"  Balanced accuracy difference: "
        f"{row['balanced_accuracy_difference']:+.4f}"
    )

    print(
        f"  Average common genes: "
        f"{row['common_genes_mean']:.2f} out of 50"
    )

    print(
        f"  Mean Jaccard similarity: "
        f"{row['jaccard_mean']:.4f}"
    )

print("\nNote:")
print(
    "These are descriptive results. Positive performance "
    "differences indicate higher average scores with leaky "
    "feature selection; they do not establish statistical "
    "significance or causation."
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "Selectors found:",
    sorted(combined["selector"].dropna().unique().tolist()),
)

print(
    "Both safe and leaky results available:",
    (
        combined["safe_macro_f1"].notna().all()
        and combined["leaky_macro_f1"].notna().all()
    ),
)

print(
    "All Jaccard means are between 0 and 1:",
    combined["jaccard_mean"].between(0, 1).all(),
)

print(f"\nOutput saved to:\n{OUTPUT_FILE}")

print("\nAnalysis completed successfully.")