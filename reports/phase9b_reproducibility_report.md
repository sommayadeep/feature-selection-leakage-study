# Phase 9B: Reproducible Final Analysis Pipeline

## Scope

This phase converted the ad hoc Phase 7 comparative analysis into the
maintained [`analyze_phase7_results.py`](../src/analyze_phase7_results.py)
workflow. No experiment was rerun, no dataset was changed, and the
experimental methodology and metric definitions were preserved.

The script reads only the shared ID3 results and the corrected Phase 6 k-NN
results. It rejects a superseded k-NN input unless its filename identifies the
`phase6_corrected` output.

## Files created or modified

Created:

- [`src/analyze_phase7_results.py`](../src/analyze_phase7_results.py)
- [`tests/test_phase9b_analysis.py`](../tests/test_phase9b_analysis.py)
- [`reports/phase9b_reproducibility_manifest.json`](./phase9b_reproducibility_manifest.json)
- This report.
- [`reports/phase9b_backup_20261024_095818/`](./phase9b_backup_20261024_095818/), containing the pre-run Phase 7 artifacts and hashes.

Modified by verified regeneration:

- The six Phase 7 CSV tables under `results/tables/`.
- The three Phase 7 PNG figures under `reports/figures/`.

The six CSVs were backed up first and reproduced exactly. The original
pre-regeneration hashes are in
[`phase9b_backup_20261024_095818/original_sha256.txt`](./phase9b_backup_20261024_095818/original_sha256.txt).

## Inputs

The maintained script validates and reads:

- `results/tables/shared_3x5_id3_results.csv`
- `results/tables/shared_3x5_id3_summary.csv`
- `results/tables/shared_3x5_knn_phase6_corrected_results.csv`
- `results/tables/shared_3x5_knn_phase6_corrected_summary.csv`
- `configs/outer_fold_manifest.csv`

It verifies the 15-fold, 48/24 train/test structure, 50 selected features,
safe/leaky pairing, finite metrics in `[0, 1]`, and consistency of each
model's saved summary with its fold-level rows. The ID3 and corrected k-NN
fold metadata are compared before analysis.

## Recovered analysis logic

The script preserves the existing definitions:

- Leakage difference: `leaky - safe`.
- Classifier difference: safe k-NN minus safe ID3 on matched folds.
- Selector contrast: MI leakage difference minus ANOVA leakage difference.
- Metrics: Macro-F1 and balanced accuracy.
- Pairing key: selector, selection mode, repeat, fold, and global fold.
- Bootstrap: 20,000 resamples, NumPy default generator, seed `20261024`,
  percentile values at 2.5% and 97.5%.

One seeded generator is consumed in the established order: eight leakage
comparisons, four classifier comparisons, and four selector contrasts. The
intervals are labeled descriptive percentile bootstrap summaries. They are
not inferential confidence intervals because repeated-CV folds are dependent.

## Exact command

The final regeneration command was:

```text
.venv/bin/python src/analyze_phase7_results.py --overwrite --manifest-output reports/phase9b_reproducibility_manifest.json
```

The manifest records the resolved interpreter command, Python version,
platform, package versions, bootstrap settings, input hashes, and output
hashes.

## Validation

Focused Phase 9B tests:

```text
.venv/bin/python -m unittest tests/test_phase9b_analysis.py
```

Result: **4 tests passed**.

Full test suite:

```text
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Result: **18 tests passed**.

The six regenerated CSVs were compared against their pre-run versions before
promotion:

- `phase7_paired_differences.csv`: exact byte match.
- `phase7_paired_differences_summary.csv`: exact byte match.
- `phase7_model_comparison.csv`: exact byte match.
- `phase7_model_comparison_summary.csv`: exact byte match.
- `phase7_selector_comparison.csv`: exact byte match.
- `phase7_selector_comparison_summary.csv`: exact byte match.

Independent dataframe validation also found equal schemas, row counts, text
fields, and numeric values for all six tables. The expected row counts are
120, 8, 60, 4, 60, and 4, respectively.

The three regenerated figures are not byte-identical to the historical PNGs.
The maintained script reconstructs the same requested plots from the same
saved data, but the original plotting session did not preserve its plotting
code or rendering metadata. The pre-run historical figures remain available
in the Phase 9B backup. The regenerated figures are publication-readable
reconstructions, not claims of pixel-level reproduction.

## Remaining limitations

- Reproducibility is established for the Phase 7 tables and their numerical
  definitions, not for the original plotting session's exact rendering.
- The analysis remains descriptive and protocol-local; it does not establish
  statistical significance or independent generalization.
- Repeated cross-validation reuses patients across folds, so the 15 fold rows
  are not independent replications.
- Dataset provenance, exact transformed-feature mapping, and licensing remain
  unresolved as documented in
  [`phase9a_dataset_provenance.md`](./phase9a_dataset_provenance.md).
- The current work does not address the external-validation gap identified in
  [`phase8_scientific_audit.md`](./phase8_scientific_audit.md).
