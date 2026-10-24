# Phase 3 Protocol Validation

## Scope and restrictions

Phase 3 integrated the shared outer-fold protocol and audited leakage
boundaries without running the full ID3, k-NN, nested-CV, or stability
experiments. Historical result tables and the Phase 2 backup were preserved.
No result values were regenerated, edited, or copied into historical outputs.

## 1. Shared outer-fold manifest

### Files inspected

- [`configs/shared_experiment_config.json`](../configs/shared_experiment_config.json)
- [`configs/outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv)
- [`src/experiment_config.py`](../src/experiment_config.py)
- [`src/run_id3_experiment.py`](../src/run_id3_experiment.py)
- [`src/run_safe_experiment.py`](../src/run_safe_experiment.py)
- [`src/compare_leakage.py`](../src/compare_leakage.py)
- [`src/run_nested_experiment.py`](../src/run_nested_experiment.py)

### Integration

`load_outer_split_manifest()` now:

1. Checks that the manifest exists and has 15 rows.
2. Checks seed, fold-count, repeat-count, and train/test sizes.
3. Regenerates the deterministic expected splits from the shared configuration
   and rejects any differing train or test indices.
4. Rejects out-of-range, duplicate, overlapping, or incomplete splits.
5. Checks that every class appears in every train and test split with
   stratified fold-sized counts.
6. Checks that every patient occurs in the test set exactly once per repeat.

The shared ID3 runner, leakage-safe k-NN runner, and matched k-NN leakage
runner all load this manifest. Their new output names contain
`shared_3x5`, so a future run cannot overwrite the historical result tables.
Safe and leaky conditions within the matched runners iterate over the same
validated split list.

### Validation evidence

The processed leukemia labels produce:

- 15 outer splits.
- 48 training and 24 test patients per split.
- Both classes represented in every train and test split.
- No train/test overlap.
- All 72 patients represented in the test sets.
- Exactly five test appearances per patient.
- Exact agreement between the saved manifest and deterministic
  `RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)`.

## 2. Leakage-boundary audit

### ID3

For each shared outer split:

- Median imputation is fitted on `X_train` and only transforms test/all
  patients afterward.
- Safe ANOVA or mutual-information selection is fitted on `X_train` and
  `y_train`.
- The deliberate leaky selector is fitted on imputed all-patient features and
  all labels, including held-out labels.
- Quantile discretization is fitted on selected training features only.
- The tree is fitted on training data only.

Thus the ID3 control is explicitly a feature-selection leakage condition, not a
claim of valid generalization performance.

### k-NN

The safe k-NN runner uses one scikit-learn pipeline containing imputation,
scaling, feature selection, and fixed `k=3` classification. Passing the shared
split list to `cross_validate` fits every pipeline step separately within each
training fold.

The matched leakage runner now fits the imputer and scaler on each training
fold before transforming all patients. It then fits the selector using all
labels, including the held-out labels, and fits the classifier on selected
training features. This keeps preprocessing order and fitted statistics
consistent with the safe path while isolating selector leakage from the prior
global-imputation confound. The current dataset has no missing values, but the
boundary is now correct for datasets that do.

The leaky path remains a diagnostic control and must not be interpreted as an
unbiased estimate of generalization.

### Nested k-NN

[`src/run_nested_experiment.py`](../src/run_nested_experiment.py) remains
separate. It performs 3-fold outer and 3-fold inner nested CV, tunes feature
count, neighbor count, and distance weighting, and has no matched leaky
counterpart. It must not be pooled with the fixed-configuration 3x5 protocol.

## 3. Historical comparability

The existing tables remain preliminary and are not silently relabeled:

- Historical ID3 tables use the same 3x5 split family and remain preserved.
- Historical preliminary k-NN tables use 3 folds x 2 repeats and remain
  incomparable with the shared 3x5 protocol.
- Nested k-NN uses a distinct model-selection design.
- Feature-stability results use separate 20-resample, 100-feature settings.

No Phase 3 runner was executed, so no new performance values are available.

## 4. Tests and checks

### Modified or created

- [`src/experiment_config.py`](../src/experiment_config.py): added strict
  manifest loading and split-integrity validation.
- [`src/run_id3_experiment.py`](../src/run_id3_experiment.py): consumes the
  shared manifest and writes only shared-protocol output names.
- [`src/run_safe_experiment.py`](../src/run_safe_experiment.py): consumes the
  shared manifest and uses shared configuration values.
- [`src/compare_leakage.py`](../src/compare_leakage.py): consumes the shared
  manifest and removes global-imputation leakage from the selector-only
  control.
- [`tests/test_id3_and_configuration.py`](../tests/test_id3_and_configuration.py):
  added real-manifest integrity, identical-model-split, and training-only
  preprocessing tests.
- [`configs/shared_experiment_config.json`](../configs/shared_experiment_config.json):
  records that the shared protocol is integrated while legacy outputs remain
  separate.
- [`reports/methodology_audit.md`](./methodology_audit.md) and
  [`reports/completion_checklist.md`](./completion_checklist.md): updated for
  the Phase 3 findings.

### Results

The lightweight test suite completed successfully:

```text
Ran 11 tests in 0.159s
OK
```

Python compilation passed for all changed Python files. Pylance reported no
diagnostics for [`src/experiment_config.py`](../src/experiment_config.py);
remaining workspace diagnostics are pre-existing unused-import/variable
warnings in unrelated legacy and backup scripts.

No full experiment was run. Historical result-table files remain byte-for-byte
unchanged, and the Phase 2 backup at
[`reports/phase2_backup_20261024_0913/`](./phase2_backup_20261024_0913) was not
modified.

## 5. Remaining decisions before Phase 4

1. Approve the shared 3x5 fixed-configuration protocol for new ID3/k-NN
   performance runs.
2. Decide whether future k-NN runs should additionally save per-fold metrics,
   selected feature sets, and patient-level predictions.
3. Decide how the separate nested-CV and feature-stability analyses should
   contribute to final claims; they should not be merged with the shared
   fixed-protocol estimates.
4. Resolve dataset provenance, biological feature-name mapping, and licensing
   limitations before writing a final paper.
