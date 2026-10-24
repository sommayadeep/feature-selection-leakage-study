# Phase 4 Controlled Experiment Report

## Scope and controls

Phase 4 executed the approved shared fixed-configuration protocol after
pre-run validation and bounded smoke tests. The historical result tables and
the Phase 2 backup were preserved. No second dataset or manuscript content was
added.

The shared protocol used:

- 72 patients and 7,128 features from the processed leukemia dataset.
- Seed `42`.
- The validated 3-fold x 5-repeat, 15-split outer manifest.
- 50 selected features.
- ID3: 3 quantile bins, maximum depth 5, minimum split size 4.
- k-NN: fixed `k=3`.
- Safe and leaky conditions paired within each outer fold.

The leaky condition is a diagnostic control only. It fits feature selection
using all labels, including held-out labels. Imputation, scaling, and ID3
discretization remain training-fold operations.

## Stage 1: pre-run validation

Before execution, the following were checked:

1. `run_id3_experiment.py`, `run_safe_experiment.py`, and
   `compare_leakage.py` consume `configs/outer_fold_manifest.csv`.
2. The manifest loader verifies deterministic indices, stratification,
   disjointness, complete coverage, and five test appearances per patient.
3. Safe ID3 fits imputation, selection, and discretization on training data.
4. Safe k-NN uses a pipeline whose imputer, scaler, and selector fit within
   each training fold.
5. The leaky k-NN control uses held-out labels only in the selector.
6. New output paths use `shared_3x5_*` names and do not overlap historical
   output names.
7. The lightweight test suite passed before experiments.

## Stage 2: smoke tests

Commands:

```text
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
  src/run_id3_experiment.py --max-folds 2 --output-stem smoke_id3

PYTHONPATH=src \
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
  src/compare_leakage.py --max-folds 2 --output-stem smoke_knn
```

Both smoke runs completed successfully. Validation confirmed:

- 8 fold-condition records per smoke run.
- No duplicate selector/mode/fold keys.
- Valid binary predictions.
- Finite Macro-F1 and balanced-accuracy values in `[0, 1]`.
- 50 selected features per record.
- Predictions and confusion records for every fold-condition.
- Saved summaries equal recomputed smoke means.

Smoke files are diagnostic artifacts only and are not used in paper claims.

## Stage 3: full shared ID3 experiment

Command:

```text
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
  src/run_id3_experiment.py --output-stem shared_3x5_id3
```

The run completed all 15 outer folds for ANOVA and mutual information under
both safe and leaky selection:

| Selector | Mode | Macro-F1 mean | Macro-F1 SD | Balanced accuracy mean | Balanced accuracy SD |
|---|---|---:|---:|---:|---:|
| ANOVA | safe | 0.871069 | 0.047607 | 0.889769 | 0.039055 |
| ANOVA | leaky | 0.880167 | 0.053263 | 0.898102 | 0.043413 |
| MI | safe | 0.862635 | 0.065936 | 0.879676 | 0.057951 |
| MI | leaky | 0.891404 | 0.059156 | 0.906991 | 0.047534 |

These are descriptive repeated-CV estimates. No significance claim is made.

## Stage 4: full shared k-NN experiment

The first direct script launch failed twice with
`ModuleNotFoundError: No module named 'pandas'`, although the configured
interpreter reported pandas installed and imports succeeded through the module
path. No output was produced by those failed launches and no dependency was
changed. The reproducible invocation explicitly set `PYTHONPATH=src`.

During final validation, the leaky k-NN implementation was also corrected
before accepting its full results: its selector now receives data transformed
by an imputer and scaler fitted on the training fold, matching the safe
preprocessing order. Held-out labels remain used only by the selector. A
corrected two-fold smoke test passed, and the 15-fold k-NN run was rerun from
scratch. The values below are from that corrected run.

Command:

```text
PYTHONPATH=src \
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
  src/compare_leakage.py --output-stem shared_3x5_knn
```

The run completed all 15 outer folds for both selectors and both conditions:

| Selector | Mode | Macro-F1 mean | Macro-F1 SD | Balanced accuracy mean | Balanced accuracy SD |
|---|---|---:|---:|---:|---:|
| ANOVA | safe | 0.954069 | 0.039085 | 0.956343 | 0.041574 |
| ANOVA | leaky | 0.959948 | 0.038593 | 0.960185 | 0.039717 |
| MI | safe | 0.960125 | 0.034251 | 0.960648 | 0.035765 |
| MI | leaky | 0.981587 | 0.023353 | 0.982176 | 0.024265 |

These values are descriptive only. The leaky rows are not valid estimates of
generalization performance.

## Stage 5: artifact validation

The following checks passed for both full experiments:

- 60 unique fold-condition records: 2 selectors x 2 modes x 15 folds.
- No duplicate selector/mode/fold records.
- Fold IDs are exactly 1 through 15.
- Metrics are finite and within `[0, 1]`.
- 60 selected-feature records, each containing 50 features.
- 1,440 prediction rows per model.
- No duplicate patient prediction within a selector/mode/fold.
- 60 confusion-matrix records per model.
- Saved summaries agree with independently recomputed fold means and standard
  deviations.
- ID3 and k-NN use the same global fold IDs.
- All 30 historical backed-up result CSVs remain byte-for-byte unchanged.

## Generated files

Required files:

- [`shared_3x5_id3_results.csv`](../results/tables/shared_3x5_id3_results.csv)
- [`shared_3x5_id3_summary.csv`](../results/tables/shared_3x5_id3_summary.csv)
- [`shared_3x5_knn_results.csv`](../results/tables/shared_3x5_knn_results.csv)
- [`shared_3x5_knn_summary.csv`](../results/tables/shared_3x5_knn_summary.csv)

Additional artifacts:

- [`shared_3x5_id3_selected_features.csv`](../results/tables/shared_3x5_id3_selected_features.csv)
- [`shared_3x5_id3_predictions.csv`](../results/tables/shared_3x5_id3_predictions.csv)
- [`shared_3x5_id3_confusion_matrices.csv`](../results/tables/shared_3x5_id3_confusion_matrices.csv)
- [`shared_3x5_knn_selected_features.csv`](../results/tables/shared_3x5_knn_selected_features.csv)
- [`shared_3x5_knn_predictions.csv`](../results/tables/shared_3x5_knn_predictions.csv)
- [`shared_3x5_knn_confusion_matrices.csv`](../results/tables/shared_3x5_knn_confusion_matrices.csv)

Smoke-test artifacts are also present with `smoke_` prefixes and are excluded
from the full-result interpretation.

## Tests and final limitations

Pre-run and final lightweight validation:

```text
Ran 11 tests
OK
```

Python compilation passed for the changed runners and test module. Remaining
Pylance diagnostics are pre-existing unused-import warnings in unrelated
legacy/backup scripts.

The results do not establish statistical significance, causal effects, or
generalization beyond this dataset. Historical k-NN tables used different
folds and remain incomparable. Nested-CV and feature-stability experiments
remain separate designs. Dataset provenance, biological feature-name mapping,
and licensing metadata are still unresolved.
