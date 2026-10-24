# Phase 7 Final Comparative Analysis

## Scope and protocol

This analysis uses existing saved outputs only. No full experiment was rerun,
no ID3 result was changed, and no historical report or result table was
overwritten.

The analyzed files use the canonical shared manifest:
[`outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv). The protocol
contains 72 patients, 7,128 features, 3 outer folds x 5 repeats, and 15
matched outer splits. Every split contains 48 training and 24 test patients;
each patient appears in a test fold five times. Both models select 50
features. ID3 uses the fixed shared tree settings and k-NN uses fixed `k=3`.

The authoritative k-NN files are the corrected Phase 6 outputs:
[`shared_3x5_knn_phase6_corrected_results.csv`](../results/tables/shared_3x5_knn_phase6_corrected_results.csv)
and its associated predictions, selected features, confusion matrices, and
summary. The earlier Phase 5 report correctly documented the pre-correction
block; this report supersedes that block for the corrected k-NN analysis.

Repeated-CV fold rows are dependent because training sets overlap across
repeats and folds. They are paired resampling observations, not 15
independent experimental replications.

## Independent artifact verification

The ID3 and corrected k-NN result tables each contain exactly 60 unique
selector/mode/fold records: 2 selectors x 2 modes x 15 folds. Global folds
are exactly 1 through 15, with matching repeat/fold metadata and 48/24
training/test sizes.

For both models, the corresponding prediction and confusion-matrix files
were independently checked:

- 1,440 prediction rows per model.
- 60 confusion-matrix rows per model.
- No duplicate patient prediction within a selector/mode/fold.
- Prediction-derived Macro-F1 and balanced accuracy match the result rows.
- Prediction-derived confusion counts match the saved confusion rows.
- Safe and leaky rows are paired by the same selector and global fold.
- ID3 and corrected k-NN rows use the same manifest metadata.

No new methodological inconsistency was found in the corrected k-NN outputs.
The corrected source uses one shared training-fitted imputer/scaler and
differs between conditions only in the labels supplied to feature selection.

## Paired leakage differences

The fold-level table is
[`phase7_paired_differences.csv`](../results/tables/phase7_paired_differences.csv).
Differences are defined as **leaky minus safe**.

The intervals are 95% percentile bootstrap intervals over the 15 paired fold
differences, using 20,000 resamples and seed `20261024`. This bootstrap treats
the observed paired rows as exchangeable resampling units. Because repeated
cross-validation folds share training observations, the intervals are
descriptive sensitivity summaries, not definitive inferential confidence
intervals and not significance tests.

| Classifier | Selector | Metric | Safe mean | Leaky mean | Mean difference | Median difference | SD | Descriptive 95% interval | Positive / zero / negative |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ID3 | ANOVA | Macro-F1 | 0.871069 | 0.880167 | 0.009097 | 0.000000 | 0.018973 | [0.000000, 0.018968] | 3 / 12 / 0 |
| ID3 | ANOVA | Balanced accuracy | 0.889769 | 0.898102 | 0.008333 | 0.000000 | 0.018551 | [0.000000, 0.018750] | 3 / 12 / 0 |
| ID3 | MI | Macro-F1 | 0.862635 | 0.891404 | 0.028769 | 0.000000 | 0.051282 | [0.005926, 0.056299] | 5 / 10 / 0 |
| ID3 | MI | Balanced accuracy | 0.879676 | 0.906991 | 0.027315 | 0.000000 | 0.051444 | [0.006667, 0.055278] | 5 / 10 / 0 |
| k-NN | ANOVA | Macro-F1 | 0.954069 | 0.959948 | 0.005878 | 0.000000 | 0.046166 | [-0.016394, 0.027795] | 5 / 7 / 3 |
| k-NN | ANOVA | Balanced accuracy | 0.956343 | 0.960185 | 0.003843 | 0.000000 | 0.046795 | [-0.019772, 0.026296] | 5 / 7 / 3 |
| k-NN | MI | Macro-F1 | 0.960125 | 0.981587 | 0.021462 | 0.000000 | 0.023776 | [0.009286, 0.033606] | 7 / 8 / 0 |
| k-NN | MI | Balanced accuracy | 0.960648 | 0.982176 | 0.021528 | 0.000000 | 0.025556 | [0.009491, 0.034259] | 7 / 8 / 0 |

The results show positive mean differences in all eight comparisons. The
observed difference is larger for MI than ANOVA within both classifiers.
These are descriptive associations produced by allowing held-out labels into
feature selection; they are not causal estimates of leakage effects and do
not establish statistical significance.

The fold-level figure is
[`phase7_fold_leakage_differences.png`](figures/phase7_fold_leakage_differences.png).

## Safe-condition classifier comparison

The fold-level table is
[`phase7_model_comparison.csv`](../results/tables/phase7_model_comparison.csv).
Differences are defined as **safe k-NN minus safe ID3** on the same selector,
outer fold, training patients, and test patients.

| Selector | Metric | ID3 safe mean | k-NN safe mean | Mean difference | Median difference | SD | Descriptive 95% interval | k-NN higher / equal / lower |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ANOVA | Macro-F1 | 0.871069 | 0.954069 | 0.083000 | 0.085131 | 0.047595 | [0.060109, 0.106454] | 14 / 1 / 0 |
| ANOVA | Balanced accuracy | 0.889769 | 0.956343 | 0.066574 | 0.062500 | 0.044455 | [0.045370, 0.088704] | 14 / 1 / 0 |
| MI | Macro-F1 | 0.862635 | 0.960125 | 0.097491 | 0.124036 | 0.072297 | [0.061920, 0.132400] | 13 / 1 / 1 |
| MI | Balanced accuracy | 0.879676 | 0.960648 | 0.080972 | 0.093750 | 0.065970 | [0.047639, 0.112500] | 13 / 1 / 1 |

On these shared folds, k-NN has higher safe-condition performance for most
folds and both selectors. This is a descriptive comparison on one dataset
with 72 patients and cannot establish universal classifier superiority.
Potential explanations such as biological structure, preprocessing
interaction, or model suitability require additional datasets and
prespecified analyses; they are not inferred here.

The matched-fold figure is
[`phase7_safe_classifier_differences.png`](figures/phase7_safe_classifier_differences.png).

## Selector comparison

The fold-level table is
[`phase7_selector_comparison.csv`](../results/tables/phase7_selector_comparison.csv).
For each classifier and fold, the comparison is:

`(MI leaky - MI safe) - (ANOVA leaky - ANOVA safe)`.

| Classifier | Metric | ANOVA leakage mean | MI leakage mean | MI minus ANOVA mean | Median | SD | Descriptive 95% interval | MI higher / equal / lower |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ID3 | Macro-F1 | 0.009097 | 0.028769 | 0.019672 | 0.000000 | 0.059587 | [-0.006853, 0.051564] | 5 / 7 / 3 |
| ID3 | Balanced accuracy | 0.008333 | 0.027315 | 0.018981 | 0.000000 | 0.058978 | [-0.006623, 0.050509] | 5 / 7 / 3 |
| k-NN | Macro-F1 | 0.005878 | 0.021462 | 0.015584 | 0.000000 | 0.055251 | [-0.011868, 0.041687] | 7 / 5 / 3 |
| k-NN | Balanced accuracy | 0.003843 | 0.021528 | 0.017685 | 0.000000 | 0.057564 | [-0.009955, 0.046065] | 7 / 5 / 3 |

MI has a larger average leakage difference than ANOVA within both models in
these saved folds. This descriptive pattern is not statistically significant
evidence of a selector interaction, especially given the dependence of the
repeated-CV rows and the small sample.

## Feature stability and overlap

The feature-selection stability protocols remain separate:

| Artifact | Resampling design | Features | ANOVA mean Jaccard | MI mean Jaccard |
|---|---|---:|---:|---:|
| [`feature_stability.csv`](../results/tables/feature_stability.csv) | 20 stratified 75/25 training-subset resamples | 100 | 0.538972 | 0.414511 |
| [`id3_feature_overlap.csv`](../results/tables/id3_feature_overlap.csv) | 15 matched shared ID3 outer folds, safe vs leaky | 50 | 0.594584 | 0.481090 |

These quantities answer different questions and use different feature counts.
They are not pooled or treated as directly comparable estimates. The observed
lower MI overlap occurs in both protocols, but this does not establish
general selector superiority, biological relevance, or feature importance.
The feature names are synthetic identifiers and no biological interpretation
is supported.

## Generated outputs

Primary requested outputs:

- [`phase7_final_comparative_analysis.md`](phase7_final_comparative_analysis.md)
- [`phase7_paired_differences.csv`](../results/tables/phase7_paired_differences.csv)
- [`phase7_model_comparison.csv`](../results/tables/phase7_model_comparison.csv)
- [`phase7_selector_comparison.csv`](../results/tables/phase7_selector_comparison.csv)
- [`phase7_safe_leaky_performance.png`](figures/phase7_safe_leaky_performance.png)
- [`phase7_fold_leakage_differences.png`](figures/phase7_fold_leakage_differences.png)
- [`phase7_safe_classifier_differences.png`](figures/phase7_safe_classifier_differences.png)

Additional machine-readable summary tables were generated alongside the
requested fold-level tables:

- `phase7_paired_differences_summary.csv`
- `phase7_model_comparison_summary.csv`
- `phase7_selector_comparison_summary.csv`

## Validation and limitations

The three requested tables were regenerated from the saved result rows, and
the report values were checked against the generated summary values. Figures
use the same saved fold rows and summary means. No full experiment was rerun.

Remaining limitations:

- Repeated-CV folds are dependent; bootstrap intervals are descriptive only.
- Deliberately leaky rows are diagnostic controls, not valid generalization
  estimates.
- Classifier comparisons are limited to one small dataset.
- Historical 3x2 k-NN, nested-CV, 20-resample stability, matched ID3 overlap,
  and shared 3x5 performance artifacts remain distinct protocols.
- Dataset provenance, biological feature mapping, and licensing metadata
  remain unresolved.
- This phase does not constitute a manuscript or a claim of statistical
  significance.
