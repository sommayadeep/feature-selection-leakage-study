# Project Inventory

## Scope and inspection date

This inventory records the state of the project inspected on 2026-10-24. The
inspection was read-only: existing experiment scripts, datasets, and result
tables were not modified, imported, or rerun. The executable ID3 experiment was
inspected as source text rather than imported.

The workspace is not a Git repository, so there is no commit identifier or
versioned baseline against which the files can be compared.

## Top-level structure

| Path | Observed contents | Phase 1 status |
|---|---|---|
| `data/raw/` | `leukemia_big.csv` | Present and inspected |
| `data/processed/` | `leukemia.csv` plus macOS `._*` metadata artifact | Present and inspected |
| `src/` | Preparation, experiment, analysis, and diagnostic Python scripts | Present and inspected |
| `results/tables/` | ID3, k-NN, nested-CV, overlap, stability, and diagnostic CSVs | Present and inspected |
| `results/figures/` | Empty | Missing empirical figures |
| `results/logs/` | Empty | Missing run logs |
| `notebooks/` | Empty | No notebook documentation |
| `paper/` | Empty | No manuscript |
| `reports/` | Created during this Phase 1 audit | Phase 1 reports present |
| `.venv/` | Existing Python 3.12.13 virtual environment | Present; not modified |
| `requirements.txt` | 113 exact package requirements | Present; provenance not documented |

The `._*` files under `data/`, `results/tables/`, and `src/` are zero-byte
macOS AppleDouble metadata artifacts, not substantive project inputs or
outputs. They should be excluded from future inventory and packaging rules.

## Dataset components

| File | Observed role | Key facts |
|---|---|---|
| `data/raw/leukemia_big.csv` | Raw orientation of the leukemia expression matrix | Shape `(7129, 72)`; first row contains labels; remaining rows contain measurements |
| `data/processed/leukemia.csv` | Sample-by-feature matrix used by experiments | Shape `(72, 7129)`; 7,128 numeric columns named `gene_0001` through `gene_7128` and one `target` column |
| `src/inspect_dataset.py` | Minimal raw-file inspection | Reports shape, first rows, columns, and total missing values only |
| `src/prepare_leukemia.py` | Raw-to-processed conversion | Transposes the raw matrix, creates synthetic feature names, validates the expected dimensions/classes, and writes the processed file |

The dataset audit is documented separately in
[`reports/dataset_audit.md`](./dataset_audit.md).

## Experiment and implementation scripts

| File | Function | Important configuration or limitation |
|---|---|---|
| `src/run_id3_experiment.py` | Main custom ID3-style safe/leaky comparison | 3-fold x 5-repeat stratified CV, 50 features, 3 quantile bins, depth 5, minimum split size 4; executes immediately at import and overwrites two result tables |
| `src/run_safe_experiment.py` | Preliminary leakage-safe k-NN experiment | 3-fold x 2-repeat CV, fixed `k=3`, 50 selected features plus all-feature baseline; scikit-learn pipeline keeps preprocessing and selection inside folds |
| `src/compare_leakage.py` | Preliminary k-NN safe versus deliberately leaky comparison | 3-fold x 2-repeat matched splits, fixed `k=3`, 50 features; global feature selection and global imputation are performed before fold evaluation |
| `src/run_nested_experiment.py` | Preliminary nested k-NN model-selection experiment | 3 outer and 3 inner folds; tunes feature count, neighbors, and weights; only safe conditions and six outer-fold rows are saved |
| `src/measure_feature_stability.py` | Feature-selection stability summary | 20 stratified 75/25 resamples, 100 features per selection; saves mean and standard deviation of pairwise Jaccard only |
| `src/analyze_id3_feature_overlap.py` | Matched-fold safe/leaky feature overlap | Reads ID3 selection records and calculates per-fold Jaccard similarity |
| `src/analyze_id3_combined_results.py` | ID3 aggregate performance/overlap table | Combines fold summaries and descriptive overlap summaries |
| `src/analyze_id3_fold_relationship.py` | Matched-fold relationship analysis | Joins performance differences and Jaccard values and prints exploratory Spearman correlations |
| `src/analyze_id3_fold_differences.py` | Per-fold difference categories and summaries | Counts positive, zero, and negative Macro-F1 differences descriptively |
| `src/diagnose_id3_mi_fold.py` | MI fold diagnostic | Imports `run_id3_experiment.py`, which can rerun and overwrite the main experiment outputs; its saved diagnostic disagrees with the standalone diagnostic |
| `src/diagnose_id3_mi_fold_backup.py` | Older duplicated MI fold diagnostic | Contains a separate custom tree implementation and is not a controlled reference implementation |
| `src/diagnose_id3_mi_fold_standalone.py` | Self-contained MI fold diagnostic | Reimplements the main logic without importing the executable experiment; its summary reports exact agreement for repeat 3, fold 2 |

## Existing result tables

| File | Rows | Observed content | Audit interpretation |
|---|---:|---|---|
| `results/tables/id3_experiment_results.csv` | 60 | 15 matched folds for each of two selectors and two selection modes | Primary ID3 fold-level table; internally complete |
| `results/tables/id3_experiment_summary.csv` | 4 | Aggregate ID3 means and standard deviations | Consistent with the fold-level table |
| `results/tables/id3_feature_selection_records.csv` | 60 | One 50-feature list per ID3 selector/mode/fold | Complete matched selection records |
| `results/tables/id3_feature_overlap.csv` | 30 | Safe/leaky per-fold Jaccard and common-feature counts | Consistent with selection records |
| `results/tables/id3_fold_relationship.csv` | 30 | Matched performance differences and Jaccard values | Descriptive only; repeated folds are dependent |
| `results/tables/id3_fold_difference_analysis.csv` | 30 | Difference categories and fold summaries | Descriptive only; no paired uncertainty estimate |
| `results/tables/id3_combined_analysis.csv` | 2 | Aggregate ID3 performance and overlap | Consistent with the primary ID3 table |
| `results/tables/feature_stability.csv` | 2 | 20-resample, 100-feature Jaccard summaries | Matches the reported approximately 0.5390 and 0.4145 values |
| `results/tables/leakage_comparison.csv` | 2 | Preliminary k-NN safe/leaky aggregate means | Contains the reported preliminary k-NN Macro-F1 values but not fold-level data |
| `results/tables/leukemia_safe_knn.csv` | 3 | Preliminary safe k-NN all-feature, ANOVA, and MI summaries | Same 3x2 CV family as `leakage_comparison.csv`, but no leaky rows |
| `results/tables/nested_cv_results.csv` | 6 | Three outer folds for each safe k-NN selector | Uses a different 3x3 nested protocol and cannot be pooled with the 3x2 results |
| `results/tables/id3_mi_fold_diagnostic_summary.csv` | 2 | Older MI repeat 3/fold 2 reproduction claims | Reports mismatched scores (`0.4965` and `0.7037`) and failed checks |
| `results/tables/id3_mi_fold_standalone_summary.csv` | 2 | Self-contained MI repeat 3/fold 2 reproduction | Reports matched genes and scores (`0.7333` safe, `0.9111` leaky) |
| `results/tables/id3_mi_fold_patient_predictions.csv` | 48 | Older diagnostic predictions for 24 test patients in each mode | Conflicts with the standalone comparison and requires provenance review |
| `results/tables/id3_mi_fold_standalone_patient_comparison.csv` | 24 | Self-contained paired patient comparison | Consistent with the standalone summary, but only one fold |

No figures, logs, feature-frequency table, final publication tables, or
analysis script named `src/analyze_final_results.py` currently exist.

## Environment and reproducibility assets

- The existing virtual environment is `.venv` and reports Python 3.12.13.
- `requirements.txt` contains exact pins for 113 installed packages, including
  NumPy, pandas, SciPy, scikit-learn, matplotlib, and seaborn.
- The dependency file does not state the Python version, operating system,
  package-manager provenance, or a command that generated it.
- There is no README, experiment protocol, reproducibility report, test
  suite, CI configuration, run manifest, data-source record, or code version
  identifier.
- Output CSVs do not include dataset hashes, code hashes, fold indices, or
  per-run timestamps.

## Missing or incomplete components

The following required project components are absent or incomplete:

1. A documented dataset source, license/provenance record, and biological
   identifier mapping.
2. A reusable, import-safe experiment entry point with explicit configuration.
3. Unit tests for the custom ID3 edge cases and leakage boundaries.
4. A single standardized outer split definition shared by ID3 and k-NN.
5. Full per-fold k-NN outputs, selected-feature records, and paired
   safe/leaky uncertainty analysis.
6. Stability feature frequencies and matched safe/leaky stability analysis.
7. Final table/figure generation from validated outputs.
8. Reproducibility protocol, README, and run instructions.
9. Manuscript and final completion report.
