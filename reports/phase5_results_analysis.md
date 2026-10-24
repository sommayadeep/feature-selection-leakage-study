# Phase 5 Results Analysis and Scientific Verification

## Scope and decision

This phase independently checked the current source code and saved artifacts
before manuscript work. No full experiment was rerun, no historical result
table was modified, and no new dataset was added.

The ID3 analysis below is verified from the current implementation and saved
shared 3x5 artifacts. The planned k-NN paired and classifier-comparison
analyses are **blocked**. The saved k-NN artifacts are internally consistent,
but the current leaky path applies an additional scaling operation after the
training-fitted scaling used for feature selection. This is not test-label
leakage, but it is not the documented selector-only contrast. The affected
k-NN results are therefore not used for paired or cross-classifier
conclusions.

## Protocol and sample counts

The shared protocol is defined in
[`shared_experiment_config.json`](../configs/shared_experiment_config.json) and
[`outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv):

- 72 patients and 7,128 feature columns.
- Class labels are `ALL=47` and `AML=25`.
- Seed 42; 3 outer folds x 5 repeats = 15 outer splits.
- Each split has 48 training patients and 24 test patients.
- Each patient occurs in a test set five times across the manifest.
- 50 selected features per fold.
- ID3: 3 quantile bins, maximum depth 5, minimum split size 4.
- k-NN: fixed `k=3`.

Repeated-CV fold rows are paired resampling observations, not 15 independent
experiments. All uncertainty intervals below are descriptive only.

## Independent implementation verification

### ID3

The current ID3 runner satisfies the intended boundaries:

- Training-fold imputation is fitted at
  [`run_id3_experiment.py:593`](../src/run_id3_experiment.py:593).
- Safe selection uses training features and training labels only at
  [`run_id3_experiment.py:631`](../src/run_id3_experiment.py:631).
- The leaky control uses all labels only in the explicitly contaminated
  selector at [`run_id3_experiment.py:653`](../src/run_id3_experiment.py:653).
- ID3 discretization is fitted on selected training features and only
  transformed onto the test features at
  [`run_id3_experiment.py:688`](../src/run_id3_experiment.py:688).
- The classifier is fitted only on training observations.

### k-NN

The safe pipeline in
[`compare_leakage.py:53`](../src/compare_leakage.py:53) contains fold-local
imputation, scaling, selection, and fixed-configuration k-NN. The leaky
selector is fitted on all labels at
[`compare_leakage.py:138`](../src/compare_leakage.py:138), while the imputer
and first scaler are fitted from the training fold.

However, the leaky path then passes the already scaled selected training and
test matrices to [`make_leaky_classifier()`](../src/compare_leakage.py:67),
which contains a second `StandardScaler`. The safe path has one scaler before
selection; the leaky path has one scaler before selection plus another scaler
after selection. The additional scaler is fit from training data, so it does
not introduce held-out-label leakage, but it changes the preprocessing
condition. This discrepancy must be corrected and the k-NN experiment rerun
before a valid safe-versus-leaky or ID3-versus-k-NN claim can be made.

### Saved-artifact integrity

Independent checks of the existing final files found:

- Each model has 60 result rows: 2 selectors x 2 modes x 15 folds.
- Each model has 1,440 prediction rows, 60 selected-feature records, and 60
  confusion-matrix records.
- There are no duplicate selector/mode/fold keys.
- All saved metrics are finite and within `[0, 1]`.
- Metrics recomputed from prediction rows match the result rows for both
  models.
- Confusion counts recomputed from prediction rows match the confusion files
  for both models.
- Both models contain global folds 1--15 with 48 training and 24 test
  observations per fold.
- The ID3 selected-feature records reproduce the saved
  [`id3_feature_overlap.csv`](../results/tables/id3_feature_overlap.csv)
  values exactly: mean Jaccard 0.594584 for ANOVA and 0.481090 for MI.
- The ID3 and k-NN selected-feature keys are matched for all 60
  selector/mode/fold records, and their selected sets are identical for all
  60 matched records.

These checks establish internal consistency and split alignment. They do not
remove the k-NN preprocessing discrepancy above.

## Verified paired ID3 analysis

The fold-level table is
[`phase5_paired_differences.csv`](../results/tables/phase5_paired_differences.csv).
Differences are defined as **leaky minus safe**. The interval is a deterministic
95% percentile bootstrap over the 15 paired rows (20,000 resamples, seed
20261024). Because repeated-CV folds overlap in training observations, this is
not a valid independent-sample confidence interval or significance test.

| Selector | Metric | Safe mean | Leaky mean | Mean difference | Median difference | SD | Descriptive 95% interval | Leaky higher / equal / lower |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ANOVA | Macro-F1 | 0.871069 | 0.880167 | 0.009097 | 0.000000 | 0.018973 | [0.000000, 0.018968] | 3 / 12 / 0 |
| ANOVA | Balanced accuracy | 0.889769 | 0.898102 | 0.008333 | 0.000000 | 0.018551 | [0.000000, 0.018750] | 3 / 12 / 0 |
| MI | Macro-F1 | 0.862635 | 0.891404 | 0.028769 | 0.000000 | 0.051282 | [0.005926, 0.056299] | 5 / 10 / 0 |
| MI | Balanced accuracy | 0.879676 | 0.906991 | 0.027315 | 0.000000 | 0.051444 | [0.006667, 0.055278] | 5 / 10 / 0 |

The mean differences are descriptive changes associated with allowing
held-out labels into feature selection. They are not unbiased estimates of a
causal leakage effect and do not establish statistical significance.

The paired fold plot is
[`phase5_id3_paired_differences.png`](figures/phase5_id3_paired_differences.png).

## Classifier comparison status

[`phase5_model_comparison.csv`](../results/tables/phase5_model_comparison.csv)
contains verified safe ID3 summaries and explicit blocked rows for k-NN. The
saved k-NN safe means observed during artifact inspection were Macro-F1
0.954069 (ANOVA) and 0.960125 (MI), with balanced accuracy 0.956343 and
0.960648, respectively. These values are not used to claim that k-NN is
better than ID3 because the k-NN safe-versus-leaky implementation contrast
failed the requested consistency audit. A corrected implementation and a
fresh controlled rerun are required before the classifier comparison can be
completed.

Even after correction, any ID3-versus-k-NN result would be descriptive and
limited to this one small dataset. It would not support a universal
classifier-superiority claim.

## Feature-selection stability

The stability artifacts use distinct protocols and were not pooled:

| Artifact | Selector | Resampling design | Features | Mean Jaccard | SD |
|---|---|---|---:|---:|---:|
| [`feature_stability.csv`](../results/tables/feature_stability.csv) | ANOVA | 20 stratified 75/25 resamples | 100 | 0.538972 | 0.050484 |
| [`feature_stability.csv`](../results/tables/feature_stability.csv) | MI | 20 stratified 75/25 resamples | 100 | 0.414511 | 0.043032 |
| [`id3_feature_overlap.csv`](../results/tables/id3_feature_overlap.csv) | ANOVA | 15 matched shared outer folds, safe vs leaky | 50 | 0.594584 | 0.061833 |
| [`id3_feature_overlap.csv`](../results/tables/id3_feature_overlap.csv) | MI | 15 matched shared outer folds, safe vs leaky | 50 | 0.481090 | 0.053428 |

The first artifact measures selection stability across independent
training-subset resamples. The second measures overlap between safe and leaky
selections within matched outer folds. The different feature counts and
resampling questions mean that neither value should be treated as a direct
replicate of the other. The artifacts support lower observed overlap for MI
than ANOVA within each listed protocol, but they do not identify biological
importance or establish that one selector is generally superior.

The protocol-separated figure is
[`phase5_feature_stability_by_protocol.png`](figures/phase5_feature_stability_by_protocol.png).
The saved stability table contains only aggregate pairwise summaries, so the
20-resample artifact cannot be reconstructed independently from its CSV alone.

## Files created and validation

Created without rerunning a full experiment:

- [`phase5_results_analysis.md`](phase5_results_analysis.md)
- [`phase5_paired_differences.csv`](../results/tables/phase5_paired_differences.csv)
- [`phase5_model_comparison.csv`](../results/tables/phase5_model_comparison.csv)
- [`phase5_id3_paired_differences.png`](figures/phase5_id3_paired_differences.png)
- [`phase5_feature_stability_by_protocol.png`](figures/phase5_feature_stability_by_protocol.png)

The analysis-generation command recomputed paired ID3 metrics, summary
statistics, deterministic descriptive bootstrap intervals, feature-overlap
checks, and publication-readable figures from the saved CSV files. No
historical result file was overwritten.

## Unresolved issues and required next step

1. Correct the k-NN leaky path so the classifier receives the same
   training-fitted preprocessing representation as the safe path, without the
   extra post-selection scaler.
2. Rerun only the corrected shared k-NN experiment into a new, clearly named
   output stem; preserve the current artifacts as historical audit evidence.
3. Revalidate paired k-NN differences and the shared ID3-versus-k-NN
   comparison after that rerun.
4. Keep the historical 3x2 k-NN, nested-CV, 20-resample stability, matched
   ID3-overlap, and shared 3x5 outputs distinct in all future reporting.
5. Dataset provenance, biological feature mapping, and licensing metadata
   remain unresolved.
