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
    / "id3_feature_selection_records.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "id3_feature_overlap.csv"
)


# ============================================================
# 2. LOAD FEATURE-SELECTION RECORDS
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("ID3 SAFE VS LEAKY FEATURE-SELECTION OVERLAP")
print("=" * 65)

print(f"Records loaded: {len(df)}")


# ============================================================
# 3. COMPARE MATCHING FOLDS
# ============================================================

results = []

group_columns = [
    "dataset",
    "selector",
    "repeat",
    "fold",
    "global_fold",
]

for group_key, group in df.groupby(group_columns):

    safe_rows = group[
        group["selection_mode"] == "safe"
    ]

    leaky_rows = group[
        group["selection_mode"] == "leaky"
    ]

    if len(safe_rows) != 1 or len(leaky_rows) != 1:
        raise ValueError(
            f"Expected exactly one safe and one leaky record "
            f"for group {group_key}"
        )

    safe_genes = set(
        safe_rows.iloc[0]["selected_genes"].split(";")
    )

    leaky_genes = set(
        leaky_rows.iloc[0]["selected_genes"].split(";")
    )

    common_genes = safe_genes & leaky_genes
    union_genes = safe_genes | leaky_genes

    jaccard = (
        len(common_genes) / len(union_genes)
        if union_genes
        else 0.0
    )

    results.append({
        "dataset": group_key[0],
        "selector": group_key[1],
        "repeat": group_key[2],
        "fold": group_key[3],
        "global_fold": group_key[4],
        "safe_gene_count": len(safe_genes),
        "leaky_gene_count": len(leaky_genes),
        "common_gene_count": len(common_genes),
        "different_gene_count": len(union_genes - common_genes),
        "jaccard_similarity": jaccard,
    })


# ============================================================
# 4. SAVE FOLD-LEVEL RESULTS
# ============================================================

overlap_df = pd.DataFrame(results)

overlap_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"\nFold comparisons: {len(overlap_df)}")

print("\nResults by selector:")
print(
    overlap_df.groupby("selector")[
        [
            "common_gene_count",
            "different_gene_count",
            "jaccard_similarity",
        ]
    ]
    .mean()
    .round(4)
)

print(f"\nSaved results to:\n{OUTPUT_FILE}")


# ============================================================
# 5. VALIDATION
# ============================================================

print("\nValidation:")

print(
    "All selections contain 50 genes:",
    bool(
        (
            (overlap_df["safe_gene_count"] == 50)
            & (overlap_df["leaky_gene_count"] == 50)
        ).all()
    ),
)

print(
    "Jaccard values are between 0 and 1:",
    bool(
        overlap_df["jaccard_similarity"]
        .between(0, 1)
        .all()
    ),
)

print("\nAnalysis completed.")