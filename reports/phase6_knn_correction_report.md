# Phase 6 k-NN Preprocessing Correction Report

## Scope

Phase 6 corrected the k-NN implementation issue identified in
[`phase5_results_analysis.md`](./phase5_results_analysis.md). No ID3 source,
results, historical k-NN results, or backups were overwritten. The corrected
experiment uses a new output stem:

`shared_3x5_knn_phase6_corrected`

## Original issue

The previous `src/compare_leakage.py` implementation used a safe
`Pipeline` containing imputation, scaling, selection, and k-NN. Its leaky
path first applied a training-fitted imputer and scaler, selected features,
and then passed those already scaled selected matrices to another pipeline
containing a second `StandardScaler` and k-NN.

Although the second scaler did not use held-out labels, it meant the two
conditions did not have the same preprocessing operations. This violated the
intended comparison, which isolates only the feature-selection label boundary.

## Code correction

[`compare_leakage.py`](../src/compare_leakage.py) now uses three explicit
helpers:

- `preprocess_fold()` fits one `SimpleImputer` and one `StandardScaler` on
  training observations only, then transforms training, test, and all-patient
  views.
- `fit_knn_condition()` receives the already preprocessed arrays and the
  feature-selection label set. It fits the selector, selects the configured
  50 features, fits the shared plain `KNeighborsClassifier` on selected
  training observations, and predicts only the test observations.
- `make_knn_classifier()` creates the same fixed `k=3` classifier for both
  conditions and contains no scaler.

The safe condition calls feature selection with `(X_train_scaled, y_train)`.
The leaky condition calls feature selection with `(X_all_scaled, y)`. Thus,
the conditions now differ only in whether held-out labels are available to
feature selection.

## Tests added and run

Added regression tests to
[`test_id3_and_configuration.py`](../tests/test_id3_and_configuration.py):

- Training-only imputation and scaling statistics for the shared k-NN
  preprocessing.
- Safe versus leaky selector label boundaries using a recording selector.
- Classifier fit and prediction row counts, confirming the classifier is
  trained only on training observations and predicts only test observations.
- Shared k-NN settings and absence of a second scaler.

Commands:

```text
PYTHONPATH=src \
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
-m unittest tests/test_id3_and_configuration.py
```

Result: **14 tests passed**.

```text
PYTHONPATH=src \
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
-m py_compile src/compare_leakage.py tests/test_id3_and_configuration.py
```

Result: compilation passed.

## Smoke test

Command:

```text
PYTHONPATH=src \
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
src/compare_leakage.py --max-folds 2 --output-stem smoke_knn_phase6
```

Smoke outputs:

- [`smoke_knn_phase6_results.csv`](../results/tables/smoke_knn_phase6_results.csv)
- [`smoke_knn_phase6_summary.csv`](../results/tables/smoke_knn_phase6_summary.csv)
- [`smoke_knn_phase6_selected_features.csv`](../results/tables/smoke_knn_phase6_selected_features.csv)
- [`smoke_knn_phase6_predictions.csv`](../results/tables/smoke_knn_phase6_predictions.csv)
- [`smoke_knn_phase6_confusion_matrices.csv`](../results/tables/smoke_knn_phase6_confusion_matrices.csv)

Smoke validation passed:

- 8 unique fold-condition rows: 2 selectors x 2 modes x 2 folds.
- 192 prediction rows.
- 8 selected-feature records and 8 confusion records.
- Exactly 50 selected features per record.
- Finite metrics in `[0, 1]`.
- Prediction-derived metrics and confusion matrices match saved records.
- Saved summary means and standard deviations match recomputation.

## Corrected full experiment

Command:

```text
PYTHONPATH=src \
"/Volumes/T7 Shield/Research/leakage-safe-feature-selection/.venv/bin/python" \
src/compare_leakage.py \
--output-stem shared_3x5_knn_phase6_corrected
```

Generated outputs:

- [`shared_3x5_knn_phase6_corrected_results.csv`](../results/tables/shared_3x5_knn_phase6_corrected_results.csv)
- [`shared_3x5_knn_phase6_corrected_summary.csv`](../results/tables/shared_3x5_knn_phase6_corrected_summary.csv)
- [`shared_3x5_knn_phase6_corrected_selected_features.csv`](../results/tables/shared_3x5_knn_phase6_corrected_selected_features.csv)
- [`shared_3x5_knn_phase6_corrected_predictions.csv`](../results/tables/shared_3x5_knn_phase6_corrected_predictions.csv)
- [`shared_3x5_knn_phase6_corrected_confusion_matrices.csv`](../results/tables/shared_3x5_knn_phase6_corrected_confusion_matrices.csv)

Independent validation passed:

- 60 unique result rows: 2 selectors x 2 modes x 15 folds.
- Global folds are exactly 1 through 15.
- 1,440 prediction rows.
- 60 selected-feature records.
- 60 confusion-matrix records.
- Exactly 50 selected features per record.
- Every fold has 48 training and 24 test observations.
- Metrics are finite and within `[0, 1]`.
- No duplicate result keys or duplicate patient predictions within a
  selector/mode/fold.
- Prediction-derived Macro-F1 and balanced accuracy values match the result
  table.
- Prediction-derived confusion counts match the confusion-matrix table.
- Saved summary means and standard deviations match recomputed values.
- Fold metadata matches `configs/outer_fold_manifest.csv` and the existing
  shared ID3 results.

## Numerical comparison with the previous k-NN artifacts

The old `shared_3x5_knn_*` files remain present and separate. The corrected
Macro-F1 and balanced-accuracy fold values are numerically identical to the
previous files (`max_abs_difference = 0.0` for both metrics), and selected
feature files are byte-for-byte identical. This is expected here because the
first scaler already standardizes the selected training columns; the second
scaler was an implementation inconsistency even though it did not change the
recorded values on this dataset.

The corrected files are the authoritative k-NN artifacts for subsequent
paired analysis because their source implementation now matches the intended
preprocessing design. This numerical equality does not retroactively make the
old implementation structurally correct.

## Post-run source inspection

After execution, the source was inspected again. The only scaler in the
condition path is the shared training-fitted scaler in `preprocess_fold()`.
The previous `Pipeline`, `clone`, `make_safe_pipeline()`, and
`make_leaky_classifier()` paths are gone. The classifier is created by
`make_knn_classifier()` and is shared by both conditions.

## Remaining limitations

- The leaky rows remain diagnostic controls, not unbiased generalization
  estimates.
- Repeated outer folds are dependent and must not be treated as 15 independent
  experiments.
- The corrected outputs support completion of the previously blocked k-NN
  paired analysis, but no significance claim is made here.
- Historical 3x2 k-NN, nested-CV, feature-stability, and prior shared-output
  protocols remain separate and must not be silently pooled.
