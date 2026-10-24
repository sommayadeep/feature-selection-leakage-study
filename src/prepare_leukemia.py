from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "leukemia_big.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "leukemia.csv"

# Read the file without treating its first row as column headers.
raw = pd.read_csv(RAW_FILE, header=None)

# The first row contains the class label for each patient.
labels = raw.iloc[0].astype(str).str.strip().to_numpy()

# Remaining rows contain gene-expression measurements.
expression = raw.iloc[1:].apply(pd.to_numeric, errors="raise")

# Convert to samples × genes.
X = expression.T.copy()

# Give each gene a reproducible feature name.
X.columns = [f"gene_{i:04d}" for i in range(1, X.shape[1] + 1)]

# Add the target labels.
X["target"] = labels

# Validate the prepared dataset.
assert X.shape[0] == 72, f"Expected 72 samples, got {X.shape[0]}"
assert X.shape[1] == 7129, f"Expected 7128 genes + target, got {X.shape[1]}"
assert X["target"].notna().all()
assert set(X["target"].unique()) == {"ALL", "AML"}
assert not X.isna().any().any()
assert np.isfinite(X.drop(columns="target").to_numpy()).all()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
X.to_csv(OUTPUT_FILE, index=False)

print("=" * 55)
print("LEUKEMIA DATASET PREPARED")
print("=" * 55)
print(f"Output file: {OUTPUT_FILE}")
print(f"Samples: {X.shape[0]}")
print(f"Genes: {X.shape[1] - 1}")
print("\nClass distribution:")
print(X["target"].value_counts())
print("\nMissing values:", X.isna().sum().sum())
print("All expression values finite:",
      np.isfinite(X.drop(columns="target").to_numpy()).all())
print("\nFirst five samples:")
print(X.iloc[:5, :6].to_string(index=False))