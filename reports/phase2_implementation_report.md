# Phase 2 Implementation Report

## Scope and restrictions

Phase 2 was limited to preservation, a targeted diagnostic investigation,
import-safety refactoring, lightweight synthetic tests, and a proposed shared
outer-fold configuration. No complete ID3 or k-NN experiment was run. No
dataset was added. No existing result table was overwritten, deleted, or
manually adjusted.

## 1. Preservation of existing work

### Backup

Before modifying source, the following backup was created:

`reports/phase2_backup_20261024_0913/`

It contains:

- Copies of all substantive `src/*.py` scripts.
- Copies of all substantive `results/tables/*.csv` files.
- `original_checksums.sha256`, the initial broad checksum record.
- `original_source_and_result_checksums.sha256`, the authoritative
  substantive-source/result checksum record used for verification.

The latter excludes macOS `._*` metadata and compiled `__pycache__` bytecode.
The only intended source checksum difference is
`src/run_id3_experiment.py`, which was refactored in this phase. All original
result-table checksums still match their pre-change values.

The import-safety test also hashes all current result tables before and after
importing the refactored module; the hashes were unchanged.

## 2. ID3 diagnostic discrepancy

### Question investigated

The existing files disagreed for the same MI repeat 3, fold 2:

| Evidence source | Safe Macro-F1 | Leaky Macro-F1 | Feature sets |
|---|---:|---:|---|
| `id3_experiment_results.csv` saved row | 0.7333333333 | 0.9111111111 | Saved |
| `id3_mi_fold_standalone_summary.csv` | 0.7333333333 | 0.9111111111 | Exact match |
| `id3_mi_fold_diagnostic_summary.csv` | 0.4965034965 | 0.7037037037 | Not checked correctly by that script |

### Controlled comparison

The investigation used only the saved processed dataset, the saved fold
records, and the saved selected-feature records. It did not import
`run_id3_experiment.py` and did not write any files.

For the saved repeat 3/fold 2 split:

- The main/standalone tree logic selected exactly the saved 50 features in both
  safe and leaky modes.
- The main/standalone logic produced the saved predictions/scores:
  - Safe: Macro-F1 `0.7333333333`, balanced accuracy `0.75`.
  - Leaky: Macro-F1 `0.9111111111`, balanced accuracy `0.9375`.
- The duplicate implementation in
  `diagnose_id3_mi_fold_backup.py` selected the same features but produced:
  - Safe: Macro-F1 `0.4965034965`, balanced accuracy `0.53125`.
  - Leaky: Macro-F1 `0.7037037037`, balanced accuracy `0.75`.
- Its predictions disagreed with the main/standalone implementation for six
  safe patients and five leaky patients.

The stale diagnostic values therefore match the duplicate backup diagnostic
implementation exactly. The difference is not feature selection or fold
assignment; it is the separate `build_tree`/`predict_*` implementation in the
backup diagnostic. The main script uses `build_id3_tree` with a
`branches` representation and a `<= 1e-12` zero-gain stopping rule. The backup
diagnostic uses a separately maintained `children` implementation and a
different stopping condition. It was not a controlled reproduction of the
original tree.

### Conclusion

For this saved fold, the reproducible result is the main/standalone result,
because it matches both the saved selected-feature records and the saved
fold-level ID3 scores. The older
`id3_mi_fold_diagnostic_summary.csv` and
`id3_mi_fold_patient_predictions.csv` remain preserved as historical,
inconsistent artifacts. They were not altered to make the project appear
consistent.

Remaining uncertainty is provenance: the workspace does not record exactly
which historical command generated the stale diagnostic files. The exact
numeric match to `diagnose_id3_mi_fold_backup.py` establishes the implementation
cause, but not the original run timestamp or environment.

## 3. Import-safety refactor

### Modified file

- [`src/run_id3_experiment.py`](../src/run_id3_experiment.py)

### Change

The data loading, embedded toy check, cross-validation loop, result writing,
summary calculation, and CLI output were moved into `main()`. Execution now
occurs only under:

```python
if __name__ == "__main__":
    main()
```

The public helper functions remain importable, and the existing constants,
default configuration, algorithm, output paths, and command-line behavior are
unchanged when the file is executed as a script. No full execution was used
to validate this claim; source compilation and focused helper tests were used.

### Verification

- Pylance syntax check: no syntax errors.
- Python compilation: passed.
- Import test: passed.
- Result-table hashes before and after import: unchanged.

## 4. Lightweight tests

### Created file

- [`tests/test_id3_and_configuration.py`](../tests/test_id3_and_configuration.py)

The eight standard-library `unittest` cases cover:

1. Empty, pure, and balanced entropy.
2. Perfect and zero information gain.
3. Pure decision-tree nodes.
4. Empty labels and exhausted feature sets.
5. Maximum-depth and minimum-sample stopping.
6. Deterministic tree construction and predictions.
7. Safe selection using training labels only on a synthetic dataset.
8. Deterministic shared splits and matched manifest rows.
9. Import safety and result-file preservation.

The test run completed successfully:

```text
Ran 8 tests in 0.009s
OK
```

The first exploratory run emitted scikit-learn warnings from a constant
synthetic feature; the test fixture was corrected to avoid that irrelevant
warning, and the final run completed without warnings.

## 5. Shared configuration and outer-fold manifest

### Created files

- [`src/experiment_config.py`](../src/experiment_config.py)
- [`configs/shared_experiment_config.json`](../configs/shared_experiment_config.json)
- [`configs/outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv)

The proposed future matched protocol uses:

- Seed: `42`.
- Repeated stratified outer CV: 3 folds x 5 repeats.
- Selected features: 50.
- ID3 discretization: 3 quantile bins.
- ID3 maximum depth: 5.
- ID3 minimum split size: 4.
- Fixed preliminary k-NN neighbor count: 3.

The manifest contains train/test row indices, repeat/fold identifiers, and
sizes for all 15 outer folds. A focused comparison showed that every manifest
split exactly matches the current ID3 script's
`RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)` splits.

This is a proposal and manifest, not a rerun. Existing 3x5 ID3 outputs,
3x2 preliminary k-NN outputs, nested-CV outputs, and 20-resample stability
outputs remain separate and are not silently combined. Integration of the
manifest into future ID3 and k-NN runners should occur only after the Phase 3
protocol is approved.

## 6. Files modified or created

### Modified

- [`src/run_id3_experiment.py`](../src/run_id3_experiment.py)
- [`reports/completion_checklist.md`](./completion_checklist.md)
- [`tests/test_id3_and_configuration.py`](../tests/test_id3_and_configuration.py)

### Created

- [`src/experiment_config.py`](../src/experiment_config.py)
- [`configs/shared_experiment_config.json`](../configs/shared_experiment_config.json)
- [`configs/outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv)
- [`tests/test_id3_and_configuration.py`](../tests/test_id3_and_configuration.py)
- [`reports/phase2_implementation_report.md`](./phase2_implementation_report.md)
- `reports/phase2_backup_20261024_0913/` and its preserved source/result files

The test file was created and then adjusted once to remove an irrelevant
constant-feature warning; its final contents are covered by the final passing
test run.

## 7. Remaining discrepancy and uncertainty

- The historical stale diagnostic CSVs remain inconsistent by design because
  they were preserved.
- The old backup diagnostic should not be used as an authoritative
  reproduction.
- The current main experiment has not been run after refactoring, as required
  by the restriction against full experiments. A later smoke test should
  execute a small synthetic or reduced path before any full run.
- The shared manifest is not yet consumed by the legacy runners.
- No full k-NN per-fold evidence or final paired uncertainty analysis exists.

## 8. Recommended next step

After approval, Phase 3 should use the shared manifest in a new, explicitly
named validation runner or a carefully backed-up refactor. It should first run
only a small smoke test on synthetic data, then validate one controlled
leukemia fold if approved, before any complete experiment. Legacy outputs
should remain read-only reference artifacts throughout.
