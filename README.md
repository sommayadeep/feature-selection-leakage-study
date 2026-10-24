# Leakage-Safe Feature Selection for ID3 and k-NN

Measuring performance inflation and feature-selection stability on a
high-dimensional leukemia expression matrix.

## Status

**Research implementation and verified analysis artifacts. Not ready for
unrestricted manuscript submission.**

The project contains a completed, reproducible shared 3-fold x 5-repeat
comparison of safe and deliberately leaky feature selection for a custom
ID3-style classifier and fixed-configuration k-NN. The current evidence is a
descriptive methodological demonstration on one 72-sample dataset. It does
not establish biological significance, clinical utility, statistical
significance, external validity, or universal classifier/selector superiority.

The exact data-use and redistribution terms for the transformed Stanford
mirror remain unresolved. The dataset is therefore excluded from this
repository. See [`data/README.md`](data/README.md) and the
[Phase 9A provenance report](reports/phase9a_dataset_provenance.md).

## Abstract

Feature selection performed before cross-validation can expose held-out
labels to the feature-selection step and inflate observed evaluation scores.
This project compares a leakage-safe pipeline, where selection is fitted on
each training fold only, with a deliberately leaky diagnostic control, where
held-out labels are supplied only to feature selection. The comparison uses
ANOVA F-test and mutual-information selection with a custom ID3-style tree and
fixed `k=3` k-NN on a 72-sample, 7,128-feature leukemia expression matrix.

The study is designed to quantify protocol-local changes in Macro-F1 and
balanced accuracy and to keep safe/leaky comparisons paired on identical
outer folds. The intentionally leaky condition is a methodological control,
not a valid deployment or model-evaluation procedure.

## Research objectives

- Demonstrate the information boundary required for leakage-safe feature
  selection.
- Quantify descriptive score changes caused by deliberately exposing held-out
  labels to ANOVA or mutual-information selection.
- Compare the observed patterns for the implemented ID3-style and k-NN
  classifiers on identical outer splits.
- Preserve selected-feature and prediction artifacts for independent checks.
- Document the limits imposed by the small sample, unresolved data provenance,
  and synthetic feature identifiers.

## Implemented methodology

### Dataset

The local dataset contains 72 observations, 7,128 numeric features, and two
classes: 47 ALL and 25 AML. The raw matrix is an exact byte-level match to
the Stanford CASI `leukemia_big.csv` mirror. The local preparation script
transposes the matrix, generates positional names `gene_0001` through
`gene_7128`, and appends the target label.

These names are not verified biological identifiers. The upstream
transformation history is documented as lost by the Stanford source, and no
sample, batch, or patient metadata are available in the project. The source
dataset is not redistributed here; access requirements are documented in
[`data/README.md`](data/README.md).

### Feature selection

- **ANOVA F-test:** scikit-learn `SelectKBest(f_classif, k=50)`.
- **Mutual information:** scikit-learn `SelectKBest` using
  `mutual_info_classif(random_state=42)`, with `k=50`.

### Classifiers

- **ID3-style classifier:** custom information-gain tree with three
  quantile-discretization bins, maximum depth 5, and minimum split size 4.
- **k-NN:** scikit-learn `KNeighborsClassifier(n_neighbors=3)`.

### Leakage-safe condition

For each outer fold, imputation and any scaling are fitted on training
observations only. Feature selection receives training observations and
training labels only. ID3 discretization is fitted on selected training
features only. The classifier is fitted on training observations and
evaluated on the held-out fold.

### Deliberately leaky condition

Imputation and scaling remain training-fitted. Feature selection receives all
transformed observations and all labels, including held-out labels, solely to
simulate feature-selection leakage. The classifier is still fitted only on
training observations. This condition is explicitly diagnostic and must not be
reported as a valid generalization estimate.

### Cross-validation and metrics

The final shared protocol uses the canonical manifest in
[`configs/outer_fold_manifest.csv`](configs/outer_fold_manifest.csv):

- stratified 3-fold cross-validation;
- 5 repeats;
- 15 matched outer folds;
- 48 training and 24 test observations per fold;
- each patient appears in a test fold five times;
- identical folds for ID3 and k-NN.

Primary metrics are Macro-F1 and balanced accuracy. Phase 7 reports
leaky-minus-safe paired differences, safe k-NN minus safe ID3 differences, and
MI-minus-ANOVA leakage contrasts. Its percentile bootstrap intervals are
descriptive sensitivity summaries, not inferential confidence intervals,
because repeated-CV folds are dependent.

## Verified results

The final shared Phase 7 summaries were regenerated byte-for-byte from the
saved Phase 6-corrected k-NN and shared ID3 result rows by
[`src/analyze_phase7_results.py`](src/analyze_phase7_results.py).

Mean leaky-minus-safe differences:

| Classifier | Selector | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|
| ID3 | ANOVA | +0.009097 | +0.008333 |
| ID3 | Mutual information | +0.028769 | +0.027315 |
| k-NN | ANOVA | +0.005878 | +0.003843 |
| k-NN | Mutual information | +0.021462 | +0.021528 |

These are descriptive results from one dataset and one fixed protocol. They
must not be presented as causal population estimates or statistical
significance findings.

The safe-condition model comparison is also descriptive: k-NN has higher
mean scores than ID3 on the matched folds for both selectors, but this does
not establish general classifier superiority.

See:

- [`reports/phase7_final_comparative_analysis.md`](reports/phase7_final_comparative_analysis.md)
- [`reports/phase8_scientific_audit.md`](reports/phase8_scientific_audit.md)
- [`reports/phase10_publication_readiness.md`](reports/phase10_publication_readiness.md)

## Repository structure

```text
configs/       Shared experiment configuration and canonical outer-fold manifest
data/          Access instructions only; raw and processed data excluded
reports/       Audits, phase reports, reproducibility manifest, and figures
results/       Saved result tables and verified analysis figures
src/           Experiment runners, diagnostics, and analysis utilities
tests/         Lightweight implementation and analysis tests
```

Historical protocols and outputs are retained where useful, but they are not
silently combined. In particular, the historical 3x2 k-NN, nested-CV,
20-resample stability, matched ID3 overlap, and shared 3x5 outputs represent
different protocols.

## Installation

The committed requirements were tested with CPython 3.12.13:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The source modules are imported through `PYTHONPATH=src`:

```bash
export PYTHONPATH=src
```

## Tests and analysis reproduction

Run the complete lightweight test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Expected result for the current project state: 18 tests pass.

To regenerate the Phase 7 tables and figures after placing an authorized
dataset locally and confirming the saved source result hashes:

```bash
PYTHONPATH=src python src/analyze_phase7_results.py \
  --overwrite \
  --manifest-output reports/phase9b_reproducibility_manifest.json
```

The script refuses to use a superseded k-NN result file, validates the shared
fold contract, records hashes and package versions, and preserves the
documented 20,000-resample bootstrap settings. Do not overwrite existing
artifacts without a backup.

The full experiments are not required to reproduce the Phase 7 analysis.
If authorized data and a new experiment run are explicitly approved, use
distinct output stems:

```bash
PYTHONPATH=src python src/run_id3_experiment.py \
  --output-stem shared_3x5_id3_reproduction

PYTHONPATH=src python src/compare_leakage.py \
  --output-stem shared_3x5_knn_reproduction
```

These commands are computational experiments and should not be run
implicitly. The existing result artifacts are preserved in `results/tables/`.

## Reproducibility records

- [`configs/shared_experiment_config.json`](configs/shared_experiment_config.json)
- [`configs/outer_fold_manifest.csv`](configs/outer_fold_manifest.csv)
- [`reports/phase9b_reproducibility_manifest.json`](reports/phase9b_reproducibility_manifest.json)
- [`reports/phase9b_reproducibility_report.md`](reports/phase9b_reproducibility_report.md)
- [`reports/completion_checklist.md`](reports/completion_checklist.md)

The Phase 9B manifest records the exact Python command, Python and package
versions, input hashes, output hashes, random seed, and bootstrap count.

## Limitations and publication status

- Only one small dataset is available; no external validation cohort exists.
- Repeated-CV folds reuse patients and are dependent.
- The exact transformed-data license, upstream feature mapping, and sample
  metadata are unresolved.
- Synthetic positional feature names do not support biological interpretation.
- The leaky pipeline is a diagnostic control, not a valid evaluation method.
- Current uncertainty intervals are descriptive, not significance tests.

The Phase 10 publication-readiness decision is **not ready for unrestricted
manuscript drafting**. A restricted methodological paper may be considered
only after data-use terms or target-venue acceptance are documented.

## Citation

There is no published paper or verified conference acceptance associated with
this repository. Until a formal citation is available, cite the repository
and the data source described in the provenance report. The underlying study
citation recorded by the source is:

> Golub, T. R. et al. (1999). Molecular classification of cancer: class
> discovery and class prediction by gene expression monitoring. *Science*,
> 286(5439), 531–537. DOI:
> <https://doi.org/10.1126/science.286.5439.531>.
