from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "id3_fold_relationship.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "id3_fold_difference_analysis.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("ID3 FOLD-LEVEL PERFORMANCE DIFFERENCE ANALYSIS")
print("=" * 70)

print(f"Total matched folds: {len(df)}")


# ============================================================
# 3. VALIDATE INPUT
# ============================================================

required_columns = [
    "selector",
    "repeat",
    "fold",
    "safe_macro_f1",
    "leaky_macro_f1",
    "macro_f1_difference",
    "balanced_accuracy_difference",
    "jaccard_similarity",
    "common_gene_count",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# ============================================================
# 4. CLASSIFY PERFORMANCE DIFFERENCES
# ============================================================

# A small tolerance avoids treating floating-point noise
# as a meaningful performance difference.

TOLERANCE = 1e-10

def classify_difference(value):
    if value > TOLERANCE:
        return "Positive"
    elif value < -TOLERANCE:
        return "Negative"
    else:
        return "Zero"


df["difference_category"] = (
    df["macro_f1_difference"]
    .apply(classify_difference)
)


# ============================================================
# 5. COUNT POSITIVE, ZERO, AND NEGATIVE FOLDS
# ============================================================

counts = (
    df.groupby(["selector", "difference_category"])
    .size()
    .unstack(fill_value=0)
)

for category in ["Positive", "Zero", "Negative"]:
    if category not in counts.columns:
        counts[category] = 0

counts = counts[["Positive", "Zero", "Negative"]]

counts["Total"] = counts.sum(axis=1)

counts["Positive_percentage"] = (
    counts["Positive"] / counts["Total"] * 100
)

counts["Zero_percentage"] = (
    counts["Zero"] / counts["Total"] * 100
)

counts["Negative_percentage"] = (
    counts["Negative"] / counts["Total"] * 100
)


# ============================================================
# 6. IDENTIFY FOLDS WITH THE LARGEST DIFFERENCES
# ============================================================

largest_increases = (
    df.sort_values("macro_f1_difference", ascending=False)
    .groupby("selector", group_keys=False)
    .head(3)
)

largest_decreases = (
    df.sort_values("macro_f1_difference", ascending=True)
    .groupby("selector", group_keys=False)
    .head(3)
)


# ============================================================
# 7. SUMMARIZE PERFORMANCE DIFFERENCES
# ============================================================

summary = (
    df.groupby("selector")
    .agg(
        folds=("macro_f1_difference", "count"),
        mean_difference=("macro_f1_difference", "mean"),
        median_difference=("macro_f1_difference", "median"),
        std_difference=("macro_f1_difference", "std"),
        minimum_difference=("macro_f1_difference", "min"),
        maximum_difference=("macro_f1_difference", "max"),
        mean_jaccard=("jaccard_similarity", "mean"),
        mean_common_genes=("common_gene_count", "mean"),
    )
    .reset_index()
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("COUNTS BY SELECTOR")
print("=" * 70)

print(counts.round(2).to_string())

print("\n" + "=" * 70)
print("PERFORMANCE DIFFERENCE SUMMARY")
print("=" * 70)

print(summary.round(4).to_string(index=False))

print("\n" + "=" * 70)
print("THREE LARGEST PERFORMANCE INCREASES PER SELECTOR")
print("=" * 70)

print(
    largest_increases[
        [
            "selector",
            "repeat",
            "fold",
            "macro_f1_difference",
            "jaccard_similarity",
            "common_gene_count",
        ]
    ].round(4).to_string(index=False)
)

print("\n" + "=" * 70)
print("THREE LARGEST PERFORMANCE DECREASES PER SELECTOR")
print("=" * 70)

print(
    largest_decreases[
        [
            "selector",
            "repeat",
            "fold",
            "macro_f1_difference",
            "jaccard_similarity",
            "common_gene_count",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved fold-level analysis to:")
print(OUTPUT_FILE)

print("\nAnalysis completed successfully.")

print(
    "\nInterpretation note: these are descriptive fold-level "
    "comparisons. Repeated cross-validation folds are dependent, "
    "so their counts are not independent statistical trials."
)