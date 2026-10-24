from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "leukemia_big.csv"

print("=" * 60)
print("LEUKEMIA DATASET INSPECTION")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Download the dataset before continuing."
    )

data = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Raw shape:", data.shape)

print("\nFirst five rows:")
print(data.head())

print("\nColumn names:")
print(data.columns[:10].tolist())

print("\nMissing values:", data.isna().sum().sum())

print("\nDataset inspection completed.")