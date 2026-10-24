# Methodology Audit

## Audit conclusion

The project contains a useful exploratory implementation of leakage-safe versus
deliberately leaky feature selection for a small high-dimensional leukemia
dataset. The main ID3 tables are internally coherent, and the preliminary
k-NN leakage table contains the reported values. The project is not yet ready
for a final scientific claim or manuscript because independent verification is
incomplete and historical experimental protocols differ across scripts. Phase
3 now provides a validated shared 3-fold x 5-repeat manifest for new directly
comparable ID3 and fixed-configuration k-NN runs without rewriting historical
tables.

The findings below are based only on inspected source and local CSV files. No
full experiment was rerun during Phase 1.

## Evidence and reproducibility assessment

### Values that are internally supported

The saved ID3 fold table has 60 rows: 15 matched outer folds for each selector
and selection mode. Recomputing group means from those rows gives:

| Selector | Safe Macro-F1 | Leaky Macro-F1 | Difference | Safe balanced accuracy | Leaky balanced accuracy |
|---|---:|---:|---:|---:|---:|
| ANOVA | 0.8711 | 0.8802 | +0.0091 | 0.8898 | 0.8981 |
| MI | 0.8626 | 0.8914 | +0.0288 | 0.8797 | 0.9070 |

The corresponding full-precision values are present in
`id3_combined_analysis.csv` and agree with the fold rows. The saved ID3
selection records contain exactly one 50-feature set for every selector,
mode, repeat, and fold, and the overlap table contains 30 matched comparisons.

The reported stability values are also present:

- ANOVA: mean Jaccard 0.5389716415.
- Mutual information: mean Jaccard 0.4145111327.

These values come from a separate 20-resample experiment selecting 100
features, not from the 50-feature ID3 matched-fold overlap analysis. The ID3
overlap means are different: approximately 0.5946 for ANOVA and 0.4811 for MI.
Those quantities must not be merged.

The saved preliminary k-NN leakage table contains the reported Macro-F1
values:

- ANOVA: safe 0.938383, leaky 0.961924.
- Mutual information: safe 0.961559, leaky 0.984329.

### Independent verification status

The full experiments were not rerun because Phase 1 explicitly defers
computationally expensive work. One self-contained diagnostic,
`src/diagnose_id3_mi_fold_standalone.py`, reproduces the saved MI repeat 3,
fold 2 gene sets and scores:

- Safe Macro-F1 0.7333 and balanced accuracy 0.7500.
- Leaky Macro-F1 0.9111 and balanced accuracy 0.9375.

The older `id3_mi_fold_diagnostic_summary.csv` reports different reproduced
scores (0.4965 and 0.7037) and failed checks for the same fold. This is a
material conflict, not a rounding difference. The old and standalone
diagnostics must not both be cited as validation evidence.

## Leakage audit

### ID3 main experiment

`src/run_id3_experiment.py` performs the following operations per outer fold:

1. Splits patients with `RepeatedStratifiedKFold`.
2. Fits median imputation on the training fold and transforms the test fold.
3. Fits safe ANOVA or mutual-information selection on training data only.
4. For the deliberate leaky condition, fits selection on all imputed samples
   and all labels, including held-out labels.
5. Fits quantile discretization on selected training features only.
6. Builds the custom ID3-style tree on training data only.
7. Predicts the held-out fold and computes Macro-F1 and balanced accuracy.

This is a reasonable isolation of feature-selection leakage for the current
ID3 comparison. The leaky condition is correctly labeled as diagnostic rather
than valid generalization evaluation. The script records the matched fold
identity and selected feature names.

Important implementation concerns:

- The Phase 2 refactor puts data loading, toy tests, the full experiment, and
  CSV writes behind `main()` and an import guard. Importing the module is now
  side-effect free.
- The historical diagnostic CSVs remain preserved and should not be used as
  interchangeable evidence.
- The tree is a custom ID3-style implementation, not a scikit-learn estimator.
  Its entropy/information-gain and fallback logic are present, but only one
  embedded toy test is provided.
- Tie behavior relies on input order (`np.argmax` and
  `Counter.most_common`) rather than an explicitly documented tie-breaking
  policy.
- The script suppresses `UserWarning` around discretization, which may hide
  information needed to diagnose degenerate bins.
- Multiple redundant assignments to `FEATURE_SELECTION_FILE` are harmless but
  indicate that the script has not been cleaned or reviewed as a stable
  research entry point.

### k-NN preliminary comparisons

`src/run_safe_experiment.py` uses a scikit-learn pipeline with imputation,
scaling, feature selection, and fixed `k=3` k-NN. These fitted operations are
inside cross-validation for the safe rows.

`src/compare_leakage.py` fits the selector per matched fold using all labels,
including held-out labels, which intentionally exposes test-label leakage.
The shared runner fits the imputer and scaler on the training fold before
transforming all patients. Safe selection uses training labels; the leaky
control uses all labels after the same training-fitted preprocessing. Thus the
shared runner isolates feature-selection leakage rather than also introducing
global preprocessing leakage. The historical aggregate k-NN table remains
preliminary and was not rewritten.

The k-NN leaky result file contains only two aggregate rows. It lacks
per-fold scores, fold IDs, selected feature lists, and patient predictions.
The aggregate difference therefore cannot be independently checked as a
paired comparison without rerunning the experiment.

### Nested k-NN experiment

`src/run_nested_experiment.py` uses safe nested CV and tunes feature count,
neighbor count, and distance weights using inner training folds. This is
methodologically preferable for model selection, but it is a separate
experiment:

- It has 3 outer folds and 3 inner folds, not the 3x2 preliminary protocol.
- It has no leaky counterpart.
- It saves six outer-fold rows, but not selected feature identities or
  predictions.
- It is not directly poolable with the other k-NN tables.

## Cross-experiment comparability

The current files mix at least three protocols:

| Analysis | Outer CV | Feature count | Model selection | Stored evidence |
|---|---|---:|---|---|
| ID3 safe/leaky | 3 folds x 5 repeats | 50 | Fixed tree settings | Per-fold scores and feature sets |
| Historical preliminary k-NN safe/leaky | 3 folds x 2 repeats | 50 | Fixed `k=3` | Aggregate means/std only |
| New shared k-NN safe/leaky runner | 3 folds x 5 repeats | 50 | Fixed `k=3` | Not run in Phase 3 |
| Nested k-NN | 3 outer folds x 3 inner folds | 10/25/50/100 tuned | Inner-grid search | Six outer-fold scores |
| Stability | 20 stratified 75/25 resamples | 100 | Selector only | Pairwise Jaccard mean/std |

The historical tables cannot be combined as if they were paired observations.
The new shared ID3 and fixed-configuration k-NN runners consume the same
validated manifest, so future results from those runners will be paired by
global fold. Nested-CV and stability analyses remain separate by design.

## Feature stability audit

The existing stability script correctly fits selection on each training
resample and computes pairwise Jaccard similarity. It does not yet satisfy the
full research requirement:

- It selects 100 features while the main ID3 comparison selects 50.
- It does not calculate individual feature selection frequencies.
- It does not compare safe and leaky feature sets under matched resamples.
- It does not save the selected sets or the resampling indices.
- It reports only two aggregate rows, making later verification difficult.

The matched ID3 overlap analysis is useful for safe-versus-leaky fold
comparison, but it is a different stability quantity from the 20-resample
table.

## Performance-difference and statistical audit

The ID3 analysis provides per-fold differences and counts of positive, zero,
and negative differences. It explicitly labels these as descriptive, which is
appropriate. The outputs do not provide:

- A paired confidence interval or dependence-aware uncertainty estimate.
- A prespecified effect-size definition.
- A justified treatment of repeated-CV dependence.
- A final confusion-matrix artifact for the complete comparison.
- A patient-level record for all ID3 folds.

The exploratory Jaccard/performance relationship is not evidence of causation.
The 30 rows also come from repeated CV and are not independent subjects.

The positive observed differences in these files support only a statement about
these particular protocols and dataset. They do not support a claim that
test-label feature selection always increases performance or that leakage
causes a general effect of a fixed magnitude.

## Unsupported or currently unsafe scientific claims

The following claims should not be made from the current project state:

1. That the reported values are fully independently reproduced; the old and
   standalone diagnostics conflict.
2. That the results generalize beyond this single dataset.
3. That selected `gene_XXXX` columns are biologically important genes.
4. That leakage produces a universal positive inflation effect.
5. That any observed difference is statistically significant.
6. That ID3 and k-NN results are directly comparable across the existing
   tables without protocol qualification.
7. That the data represent independent patients in the clinical sense; the
   workspace lacks metadata to verify this.

## Reproducibility risks

1. No Git repository, commit, code hash, run manifest, or timestamped run
   metadata exists.
2. No README or experimental protocol documents commands, versions, seeds,
   folds, or output provenance.
3. The executable ID3 script has import side effects and overwrites outputs.
4. Diagnostic scripts duplicate implementation logic and produce conflicting
   evidence.
5. Output tables omit fold indices for k-NN and dataset/code hashes.
6. There are no unit tests beyond an inline toy check.
7. There are no logs or figures to audit the runs.
8. `requirements.txt` is a broad exact environment freeze but does not state
   the Python version or how it was generated.

## Files that will require modification in later phases

No existing files were modified during Phase 1. Subject to approval, the
following are the likely change surfaces:

- `src/run_id3_experiment.py`: make import-safe, parameterized, and reusable;
  preserve the current script/output before changing it.
- `src/compare_leakage.py` and `src/run_safe_experiment.py`: standardize split
  manifests, isolate leakage types, and save per-fold evidence.
- `src/run_nested_experiment.py`: align with the final protocol or clearly
  separate it as a model-selection analysis.
- `src/measure_feature_stability.py`: align feature count/protocol and save
  frequencies and selected sets.
- `src/diagnose_id3_mi_fold.py` and
  `src/diagnose_id3_mi_fold_backup.py`: retire or quarantine duplicate
  implementations after a controlled comparison; do not silently overwrite
  current outputs.
- New test files for the ID3 implementation and leakage boundaries.
- New `README.md`, `reports/experimental_protocol.md`, and
  `reports/reproducibility_report.md`.
- New `src/analyze_final_results.py` and final tables/figures after validation.

The raw and processed dataset files and the existing result tables should be
preserved. Any regenerated outputs should use new names or an explicitly
approved backup strategy.

## Recommended next phase

1. Obtain approval for a targeted reproducibility pass, not a full rerun.
2. Copy or checksum the current outputs before any script capable of writing
   them is executed.
3. Resolve the MI fold diagnostic conflict using the standalone logic as a
   comparison baseline, without importing the executable ID3 script.
4. Create a shared, explicit outer-split manifest and a safe reusable pipeline.
5. Add lightweight unit tests for entropy, information gain, tree stopping,
   tie behavior, unseen categories, and safe/leaky selector fit scope.
6. Recompute a small smoke test first, then request approval before expensive
   full experiments.
7. Only after the evidence is consistent, generate final tables/figures and
   documentation.
