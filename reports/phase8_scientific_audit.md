# Phase 8 Scientific and Statistical Audit

## Executive verdict

**NOT READY for a conference manuscript or final scientific claims.**

The implemented shared experiments and saved artifacts are internally
consistent and support a carefully bounded methodological result: under this
fixed protocol and this one 72-patient dataset, using held-out labels during
feature selection is associated with higher observed test-fold performance
than the safe condition. The evidence does not support causal, clinical,
biological, universal classifier-superiority, or population-generalization
claims.

The minimum blocking issue is unresolved dataset provenance and licensing.
The analysis is also not fully reproducible from a committed Phase 7
generator script, and the current repeated-CV uncertainty summaries are
descriptive rather than inferential. The recommendation is to resolve
provenance and preserve a rerunnable analysis workflow before writing the
paper.

## Audit scope and validation performed

Inspected:

- [`phase5_results_analysis.md`](./phase5_results_analysis.md)
- [`phase6_knn_correction_report.md`](./phase6_knn_correction_report.md)
- [`phase7_final_comparative_analysis.md`](./phase7_final_comparative_analysis.md)
- [`methodology_audit.md`](./methodology_audit.md)
- [`dataset_audit.md`](./dataset_audit.md)
- [`completion_checklist.md`](./completion_checklist.md)
- [`shared_experiment_config.json`](../configs/shared_experiment_config.json)
- [`outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv)
- [`test_id3_and_configuration.py`](../tests/test_id3_and_configuration.py)
- [`run_id3_experiment.py`](../src/run_id3_experiment.py)
- [`compare_leakage.py`](../src/compare_leakage.py)
- [`experiment_config.py`](../src/experiment_config.py)
- [`prepare_leukemia.py`](../src/prepare_leukemia.py)
- Final ID3 and corrected k-NN result, summary, prediction, selected-feature,
  and confusion-matrix CSVs.

No dedicated Phase 7 analysis generator script was present in the project
tree; the Phase 7 output tables and figures are present, but the generation
logic was executed as an ad hoc command rather than preserved as a rerunnable
project file.

Validation performed without rerunning experiments:

- Ran the lightweight suite: **14 tests passed**.
- Independently checked both final model artifact sets:
  - 60 result rows.
  - 1,440 predictions.
  - 60 confusion rows.
  - 15 global folds.
  - 48/24 train/test sizes.
  - No duplicate fold-condition keys.
  - Prediction-derived Macro-F1 and balanced accuracy match saved results.
  - Prediction-derived confusion counts match saved confusion tables.
- Confirmed the manifest contains 15 deterministic stratified folds and each
  patient appears in the test set five times.

## 1. Experimental-design audit

### Safe feature selection

**VERIFIED.** The safe ID3 branch fits ANOVA or mutual-information selection
on `X_train_imputed, y_train` only at
[`run_id3_experiment.py:631`](../src/run_id3_experiment.py:631). The safe
k-NN branch passes `X_train_scaled, y_train` to the selector at
[`compare_leakage.py:159`](../src/compare_leakage.py:159).

The saved safe rows therefore represent selectors fitted using training
labels only, subject to the limitation that unknown patient/batch
relationships cannot be assessed from the available metadata.

### Deliberately leaky condition

**VERIFIED.** The ID3 leaky selector receives all imputed observations and
all labels at [`run_id3_experiment.py:663`](../src/run_id3_experiment.py:663).
The corrected k-NN leaky selector receives all transformed observations and
all labels at [`compare_leakage.py:170`](../src/compare_leakage.py:170).
This is the explicitly intended feature-selection contamination.

**VERIFIED.** The held-out observations are not used to fit either classifier.
ID3 builds its tree with `X_train_discrete, y_train` at
[`run_id3_experiment.py:724`](../src/run_id3_experiment.py:724). k-NN fits
with selected training columns and `y_train` at
[`compare_leakage.py:99`](../src/compare_leakage.py:99). Held-out labels are
used for evaluation and, in the leaky condition only, selector fitting.

The leaky rows are diagnostic controls, not valid unbiased estimates of
generalization performance.

### Learned preprocessing

**VERIFIED.** ID3 median imputation is fitted on the training fold and only
transformed onto test/all-patient views at
[`run_id3_experiment.py:593`](../src/run_id3_experiment.py:593). Its
quantile discretizer is fitted on selected training features and transforms
the test features at [`run_id3_experiment.py:688`](../src/run_id3_experiment.py:688).

**VERIFIED.** Corrected k-NN imputation and scaling are fitted once on
training observations and used to transform training, test, and all-patient
views at [`compare_leakage.py:59`](../src/compare_leakage.py:59). Both
conditions then use the same plain fixed-`k=3` classifier, with no second
scaler.

Transforming all observations with training-fitted parameters does not itself
fit on held-out data. In the leaky branch, the resulting all-patient feature
matrix is intentionally supplied to the selector together with all labels.

### Shared splits

**VERIFIED.** [`experiment_config.py`](../src/experiment_config.py:122)
regenerates and validates the manifest against the fixed seed and
`RepeatedStratifiedKFold` design. Validation checks row count, index range,
train/test disjointness, complete coverage, class presence, stratification,
and five test appearances per patient. The ID3 and corrected k-NN result
tables use global folds 1–15 with identical repeat/fold and train/test
metadata.

### Metrics and saved artifacts

**VERIFIED.** Macro-F1 and balanced accuracy for both models were recomputed
from saved predictions and matched the result tables. Confusion matrices were
also recomputed and matched. Saved summary means and standard deviations were
recomputed from the fold-level rows in Phase 4–7 validation.

### Feature-selection and hyperparameter documentation

**VERIFIED.** The shared configuration and source document:

- 50 selected features.
- ANOVA via `f_classif`.
- Mutual information via `mutual_info_classif` with seed 42.
- Median imputation.
- k-NN `k=3`.
- ID3 quantile discretization with 3 bins.
- ID3 maximum depth 5.
- ID3 minimum split size 4.
- Seed 42 and 3 folds x 5 repeats.

**RECOMMENDATION.** Preserve the exact software versions, command lines,
source revision/checksums, and analysis-generation script in a run manifest.
The current `requirements.txt` records pinned packages, but a complete
experiment manifest and saved Phase 7 generator are not present.

### Scope of leakage claims

**VERIFIED.** The project distinguishes deliberate feature-selection leakage
from preprocessing leakage in the corrected runners and reports. The
comparison does not prove that the dataset is free of batch, duplicate,
related-sample, or technical target leakage.

**UNRESOLVED.** There are no patient IDs, collection batches, family/group
variables, laboratory metadata, or source-level sample annotations. Unknown
technical leakage cannot be ruled out.

## 2. Statistical-claims audit

### Dependence of repeated cross-validation rows

**VERIFIED.** The Phase 7 report explicitly states that the 15 folds are
dependent paired resampling observations, not 15 independent experimental
replications. Patients recur in test sets across repeats and training sets
overlap across folds and repeats.

### Bootstrap intervals

**VERIFIED.** Phase 7 uses percentile bootstrap intervals over the 15 paired
fold differences, with 20,000 resamples and seed 20261024. The report calls
them descriptive sensitivity summaries and explicitly says they are not
definitive inferential confidence intervals or significance tests.

**CRITICAL LIMITATION.** The bootstrap resamples fold rows as if they were
exchangeable units. That is a transparent descriptive summary, but it does
not model the repeated-CV cluster structure. The intervals must not be
presented as population-level confidence intervals.

### Significance claims

**VERIFIED.** The current Phase 5–7 reports make no unsupported p-value or
statistical-significance claim. They explicitly reject significance
interpretation and limit classifier comparisons to descriptive results.

### Defensible future statistical approach

**RECOMMENDATION.** Before making inferential claims, pre-specify one primary
metric and one primary comparison, then use a dependence-aware analysis. A
reasonable sensitivity strategy is to summarize differences within each of
the five repeats and report repeat-level effects, with a cluster-aware
bootstrap over repeats and fold-within-repeat summaries. If a formal test is
required, use a clearly documented corrected repeated-CV method or a
hierarchical model that accounts for repeat/fold clustering, and report its
assumptions. With only five repeats and one dataset, effect sizes and
external replication are more defensible than a nominal p-value. No such
analysis should be implemented or claimed without a prespecified decision.

## 3. Dataset provenance and licensing audit

### What is directly documented

**VERIFIED.** The local raw file is `data/raw/leukemia_big.csv`, with no
source citation, DOI, accession, contributor, download record, or license
record in the project documentation. The preparation script reads the first
raw row as 72 class labels, transposes the remaining 7,128 numeric rows, and
assigns generated names `gene_0001` through `gene_7128` at
[`prepare_leukemia.py:11`](../src/prepare_leukemia.py:11).

The local audit verifies 72 observations, 7,128 features, classes `ALL=47`
and `AML=25`, no missing values, and finite numeric features. File hashes
were recorded in [`dataset_audit.md`](./dataset_audit.md).

### What cannot be verified

**UNRESOLVED.** The original dataset source, publication citation, accession,
license, permitted research use, sample identities, collection protocol,
preprocessing history, normalization, batch information, and feature mapping
are not present in the workspace.

**VERIFIED.** `gene_0001` through `gene_7128` are synthetic positional names
created by the local preparation script. They are not verified biological
identifiers. No gene-level biological interpretation, pathway claim, or
selected-gene claim is justified.

**RECOMMENDATION.** Resolve the authoritative source and license before
publication. Preserve the original citation/accession, download date or
provenance record, feature-identifier mapping, preprocessing description, and
permission for redistribution or publication of derived artifacts.

## 4. Generalizability audit

**VERIFIED.** The conclusions are limited to one small high-dimensional
dataset: 72 observations, 7,128 features, and a 47/25 class imbalance. There
is no independent external cohort in the workspace. Repeated cross-validation
does not create new patients or independent biological replications.

**RECOMMENDATION.** An independent external dataset would test transportability
of the safe-versus-leaky performance pattern, robustness to different
feature distributions and class proportions, and whether the apparent
classifier ordering persists outside this dataset. A second dataset is not
needed to verify the arithmetic of the current experiment, but it is needed
for claims beyond this dataset.

## 5. Claims supported by current evidence

The following claims are **VERIFIED as protocol-local descriptive claims**:

1. The safe and corrected leaky conditions use matched shared outer folds.
2. Safe feature selection and learned preprocessing are fitted using training
   observations/labels under the documented implementation.
3. The deliberate leaky condition exposes feature selection to held-out labels
   while keeping classifier fitting on training observations.
4. On the saved shared 3x5 folds, the leaky condition has higher mean
   Macro-F1 and balanced accuracy for all four classifier/selector
   combinations.
5. On these same folds, corrected k-NN has higher safe-condition mean
   performance than ID3 for both selectors.
6. MI shows a larger observed leakage difference than ANOVA within both
   classifiers in this protocol.
7. The historical 100-feature stability and 50-feature ID3 overlap artifacts
   measure different quantities and must remain separate.

## 6. Claims that must not be made

The project must not claim that:

- The observed effects are statistically significant.
- The 15 folds are 15 independent experiments or 15 independent patients.
- The leakage differences are causal estimates applicable to other studies.
- k-NN is universally or clinically superior to ID3.
- The results generalize to other leukemia cohorts, platforms, laboratories,
  or clinical populations.
- Synthetic `gene_####` names identify biologically meaningful genes.
- The dataset has known licensing, clinical provenance, or independent-patient
  status when those facts are not documented.
- The deliberate leaky rows are valid generalization estimates.

## 7. Prioritized minimum additional work

1. **Resolve provenance and licensing first.** Identify the authoritative
   source, citation/accession, license, preprocessing history, and feature
   mapping. If this cannot be resolved, restrict the paper to a transparent
   methodological demonstration and disclose the limitation prominently.
2. **Preserve a rerunnable analysis workflow.** Save the Phase 7 generation
   code, exact commands, package/interpreter versions, source checksums, and
   input/output checksums in a run manifest. Do not rely on an ad hoc shell
   command for the final tables and figures.
3. **Pre-specify the statistical reporting plan.** Choose the primary metric,
   define the paired contrast, and decide whether the study will remain
   descriptive. If inferential analysis is required, select a dependence-aware
   method before further interpretation.
4. **Obtain metadata if available.** Check patient identity, duplicate or
   related samples, batch, platform, and collection variables for hidden
   group leakage.
5. **Add an external dataset only if broader claims are intended.** Keep its
   preprocessing, mapping, and evaluation protocol separate and prespecified.
6. **Add feature-selection frequencies only if biological/stability claims
   are intended.** The current aggregate Jaccard tables do not support
   feature-level biological conclusions.

## Recommendation

**Resolve provenance and reproducibility gaps first.** Do not begin the
conference manuscript or make final biological/generalization claims yet.
After provenance/licensing are documented and the analysis workflow is made
rerunnable, the current results are suitable for a narrowly scoped
descriptive methodological paper. An external dataset should be added before
any claim of robustness or transportability beyond this cohort.
