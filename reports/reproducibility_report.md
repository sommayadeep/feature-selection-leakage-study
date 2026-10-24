# Reproducibility Report

**Project:** Feature Selection Leakage Study  
**Research title:** *Leakage-Safe Feature Selection for ID3 and k-NN:
Measuring Performance Inflation and Feature Stability on High-Dimensional
Genomic Data*  
**Repository:** <https://github.com/sommayadeep/feature-selection-leakage-study>  
**Audited branch:** `main` at commit `50e17a96c940aebc8d2c07b4daa78b2533b09846`  
**Audit scope:** Repository documentation, source, tests, configurations,
saved results, figures, and prior audit reports. No experiment was rerun and
no restricted dataset was downloaded or published for this report.

Evidence labels used below:

- **VERIFIED:** directly supported by current files or recorded command
  output.
- **UNRESOLVED:** the repository does not contain enough evidence to establish
  the claim.
- **RECOMMENDATION:** a proposed improvement, not an observed project fact.

## A. Executive summary

This project studies how feature-selection leakage changes observed
classification performance. The safe condition fits feature selection using
training observations and training labels only. The intentionally leaky
condition supplies all labels, including held-out labels, to feature
selection while continuing to fit the classifier on training observations.
The leaky condition is a diagnostic control, not a valid deployment or
generalization procedure.

The current shared comparison implements:

- ANOVA F-test feature selection.
- Mutual-information feature selection.
- A custom ID3-style decision tree.
- Fixed `k=3` k-NN.
- Macro-F1 and balanced accuracy.
- A shared stratified three-fold, five-repeat outer-fold manifest.

**VERIFIED:** The current repository contains import-safe shared runners,
configuration and manifest files, saved result tables, a maintained Phase 7
analysis generator, a maintained plotting script, focused tests, and a
GitHub Actions workflow. The latest recorded GitHub Actions run for commit
`50e17a96c940aebc8d2c07b4daa78b2533b09846` completed successfully; the local
suite has 21 tests.

**VERIFIED:** The shared Phase 7 outputs support a bounded descriptive
statement: on this 72-sample dataset and fixed protocol, all eight saved
leaky-minus-safe mean comparisons are positive. For example, the mean
differences are `+0.009097` for ID3/ANOVA Macro-F1, `+0.028769` for ID3/MI
Macro-F1, `+0.005878` for k-NN/ANOVA Macro-F1, and `+0.021462` for
k-NN/MI Macro-F1. These values come from
`results/tables/phase7_paired_differences_summary.csv` and are descriptive,
not significance findings.

**UNRESOLVED:** The exact transformed dataset's licensing, upstream
transformation history, original feature mapping, sample metadata, and
independent-patient status are not established. The project is therefore
appropriate for code sharing and conditional authorized reproduction, but not
for unrestricted biological or generalization claims.

## B. Research objectives and scope

### Central question

The project asks:

> How does using held-out labels during feature selection affect observed
> classification performance, and how do the observed differences vary between
> ANOVA and mutual-information feature selection for ID3 and k-NN?

The implementation operationalizes this question by comparing safe and
deliberately leaky selector fitting on identical outer folds. The leakage
contrast is defined as:

```text
leaky score - safe score
```

The classifier comparison is defined as:

```text
safe k-NN score - safe ID3 score
```

The selector contrast is defined as:

```text
MI leakage difference - ANOVA leakage difference
```

These definitions are documented in
`reports/phase7_final_comparative_analysis.md` and implemented in
`src/analyze_phase7_results.py`.

### Intended contribution

**VERIFIED:** The intended contribution is a methodological demonstration of
why feature selection must be fitted within the training boundary of each
evaluation fold, together with a comparison of the observed protocol-local
effect for two selectors and two classifiers.

**UNRESOLVED:** Scientific novelty relative to the literature has not been
established in this repository. No target venue, peer review, acceptance, or
publication is documented.

### Supported scope

The current evidence is limited to:

- One 72-observation, 7,128-feature leukemia matrix.
- One fixed shared outer-fold protocol.
- The implemented ID3-style tree and fixed `k=3` k-NN.
- The implemented ANOVA and mutual-information selectors.
- Descriptive paired fold comparisons.

The results do not establish biological significance, clinical utility,
population-level causal effects, universal classifier superiority, or
generalization to other datasets.

## C. Dataset provenance and access

### Known identity and source

**VERIFIED:** The local raw file was recorded as byte-for-byte identical to
the Stanford CASI mirror:

- Data page:
  <https://hastie.su.domains/CASI_files/DATA/leukemia.html>
- Direct file:
  <https://hastie.su.domains/CASI_files/DATA/leukemia_big.csv>

The immediate public source is documented in
`reports/phase9a_dataset_provenance.md`. The recorded raw-file SHA-256 is:

```text
35e84928da625da0787efb31a451dedbdf390e821a94ef74b7b7ab6cab9466d4
```

The local processed-file SHA-256 is:

```text
21ea525a9a66b19945d2f7d2788c9d83373eaafd4ffdafe00a3f959a6b677fce
```

The associated publication citation recorded in the provenance report is:

> Golub, T. R., Slonim, D. K., Tamayo, P., Huard, C., Gaasenbeek, M.,
> Mesirov, J. P., Coller, H., Loh, M. L., Downing, J. R., Caligiuri, M. A.,
> Bloomfield, C. D., & Lander, E. S. (1999). Molecular classification of
> cancer: class discovery and class prediction by gene expression monitoring.
> *Science*, 286(5439), 531–537.
> DOI: <https://doi.org/10.1126/science.286.5439.531>.

This citation establishes the publication associated with the mirror. It does
not by itself establish that the local transformed matrix is the original
supplementary data or that it may be redistributed.

### Local preparation

**VERIFIED:** `src/prepare_leukemia.py`:

1. Reads `data/raw/leukemia_big.csv` without a header.
2. Treats its first row as the 72 class labels.
3. Converts the remaining 7,128 rows to numeric values.
4. Transposes the matrix to samples by features.
5. Names columns `gene_0001` through `gene_7128`.
6. Appends the `target` column.
7. Validates 72 samples, 7,128 features, labels `ALL` and `AML`, no missing
   values, and finite features.
8. Writes `data/processed/leukemia.csv`.

The preparation script does not reconstruct the upstream numeric
transformation. The raw and processed data are intentionally excluded from
the repository by `.gitignore` and `data/README.md`.

### Provenance gaps

**VERIFIED from `reports/dataset_audit.md` and
`reports/phase9a_dataset_provenance.md`:**

- 72 observations and 7,128 numeric features.
- 47 `ALL` and 25 `AML` labels.
- No patient identifiers, collection-site metadata, batch variables,
  family/relatedness information, or duplicate-sample keys.
- Feature names are generated positional names, not verified biological
  identifiers.

**UNRESOLVED:**

- The exact upstream accession for the transformed 7,128-feature matrix.
- The upstream normalization, transformation, background correction, and
  feature-filtering history.
- The original probe/gene identifiers and their order.
- Whether all columns represent independent clinical patients.
- Whether technical batches or related samples correlate with the target.
- A license or explicit research-use and redistribution statement for the
  exact Stanford mirror.

The provenance report also documents that OpenML dataset 1104 is related but
not byte-identical to the local matrix. Its `Public` metadata must not be
treated as a license for the Stanford file.

### Authorized access procedure

`data/README.md` instructs an authorized researcher to:

1. Obtain the source file through an authorized channel.
2. Place it at `data/raw/leukemia_big.csv`.
3. Verify its SHA-256 against the provenance report.
4. Run:

   ```bash
   PYTHONPATH=src python src/prepare_leukemia.py
   ```

This is an access and preparation procedure, not a grant of permission.
**UNRESOLVED:** The repository does not establish permission to download,
redistribute, or publish the original or derived matrix.

## D. Environment and dependencies

### Python and packages

**VERIFIED:** `requirements.txt` pins the direct runtime dependencies:

```text
matplotlib==3.11.2
numpy==2.5.4
pandas==3.0.6
scikit-learn==1.9.1
```

The repository README records CPython 3.12.13 as the tested local
environment. The Phase 9B manifest records:

- CPython `3.12.13`
- NumPy `2.5.4`
- pandas `3.0.6`
- Matplotlib `3.11.2`
- scikit-learn `1.9.1`
- SciPy `1.18.1`

SciPy is a transitive dependency recorded in the manifest; it is not directly
listed in `requirements.txt`.

### CI environment

**VERIFIED:** `.github/workflows/tests.yml` uses:

- `ubuntu-latest`
- Python `3.12`
- `actions/checkout@v4`
- `actions/setup-python@v5`
- `contents: read` permissions
- A 10-minute job timeout

It installs `requirements.txt`, runs `pip check`, compiles `src` and `tests`,
and runs the unittest suite with `PYTHONPATH=src`.

The latest recorded run for the current `main` commit succeeded:
<https://github.com/sommayadeep/feature-selection-leakage-study/actions/runs/40066924702>.

### Operating-system limitations

**VERIFIED:** The local manifest was generated on macOS arm64, while CI runs
on a GitHub-hosted Ubuntu runner. The repository has evidence of a successful
CI run on Ubuntu and local validation on macOS.

**UNRESOLVED:** No equivalent validation is recorded for Windows or other
operating systems. Exact floating-point or plotting-byte behavior across
platforms is not established.

## E. Repository structure

The following substantive directories and files exist in the current
repository:

- `src/`: experiment runners, model implementation, configuration utilities,
  diagnostics, result analysis, dataset preparation, and plotting.
  Important maintained entry points are:
  - `src/run_id3_experiment.py`
  - `src/compare_leakage.py`
  - `src/experiment_config.py`
  - `src/analyze_phase7_results.py`
  - `src/plot_research_results.py`
- `tests/`: standard-library `unittest` tests for ID3 behavior, leakage
  boundaries, configuration, Phase 7 analysis, and plotting transformations.
- `configs/`: `shared_experiment_config.json` and
  `outer_fold_manifest.csv`.
- `results/tables/`: historical, diagnostic, corrected, shared-protocol, and
  analysis CSV artifacts. The files use different protocols and must not be
  pooled without checking their metadata.
- `reports/`: methodology, dataset, provenance, scientific, reproducibility,
  and publication-readiness reports.
- `reports/figures/`: historical Phase 5/7 figures plus the current
  `leakage_comparison.png`, `model_comparison.png`, and
  `feature_stability.png`.
- `data/`: access instructions only. The raw and processed datasets are not
  tracked.
- `.github/workflows/tests.yml`: automated CI workflow.

**VERIFIED:** No notebooks or manuscript files are required by the current
repository structure. The project retains historical artifacts, but they are
not automatically treated as current shared-protocol results.

## F. Experimental protocol

### Shared configuration

`configs/shared_experiment_config.json` records:

| Setting | Value |
|---|---:|
| Seed | 42 |
| Outer folds | 3 |
| Repeats | 5 |
| Selected features | 50 |
| ID3 discretization bins | 3 |
| ID3 maximum depth | 5 |
| ID3 minimum split size | 4 |
| k-NN neighbors | 3 |

`configs/outer_fold_manifest.csv` contains 15 deterministic rows with 48
training and 24 test indices per fold.

**VERIFIED in `src/experiment_config.py`:** The manifest loader regenerates
the expected `RepeatedStratifiedKFold` splits from the labels and seed, then
checks row count, metadata, train/test sizes, index equality, disjointness,
complete coverage, class presence, stratification, and repeated test coverage.

### Feature selection

**VERIFIED in `src/run_id3_experiment.py` and
`src/compare_leakage.py`:**

- ANOVA uses `SelectKBest(f_classif, k=50)`.
- Mutual information uses `SelectKBest(mutual_info_classif, k=50)` with
  `random_state=42`.

### Classifiers

- ID3 is a custom information-gain tree implemented in
  `src/run_id3_experiment.py`.
- k-NN is scikit-learn `KNeighborsClassifier(n_neighbors=3)` in the corrected
  shared runner.

### Preprocessing and information boundaries

**VERIFIED from the current source:**

- ID3 median imputation is fitted on `X_train_raw`, then applied to test and
  all-patient views.
- ID3 safe selection receives only `X_train_imputed, y_train`.
- ID3 leaky selection receives all imputed observations and all labels.
- ID3 quantile discretization is fitted on selected training features only.
- The ID3 tree is fit on selected, discretized training observations only.
- Corrected k-NN imputation and standardization are fitted on training
  observations only.
- Corrected k-NN safe selection receives training features and labels.
- Corrected k-NN leaky selection receives all transformed observations and all
  labels.
- Both k-NN classifiers are fitted only on selected training rows and
  `y_train`.

Thus the deliberate contamination is held at the feature-selection label
boundary in the corrected shared runners. This does not rule out unknown
technical or patient-level leakage in the source dataset.

### Metrics and separate protocols

The shared Phase 7 analysis uses Macro-F1 and balanced accuracy. It stores
per-fold results, predictions, selected features, and confusion matrices for
the shared ID3 and corrected k-NN runs.

The repository also contains separate experiments:

- Historical k-NN leakage comparison: 3 folds × 2 repeats.
- Nested k-NN CV: 3 outer folds × 3 inner folds, with tuned feature count,
  neighbor count, and distance weights.
- Feature stability: 20 stratified 75/25 training-subset resamples selecting
  100 features.

These are not interchangeable with the shared 3-fold × 5-repeat, 50-feature,
fixed-`k=3` comparison.

## G. Reproduction instructions

The commands below are taken from current README instructions and inspected
CLI definitions. They are not claims that a full experiment was rerun for
this report.

### Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

The Python executable must be available as `python3.12`; the repository does
not install Python itself.

### Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
python -m pip check
python -m compileall -q src tests
```

These commands do not require the restricted dataset. The current test suite
contains 21 tests.

### Dataset preparation

After authorized acquisition and SHA-256 verification:

```bash
PYTHONPATH=src python src/prepare_leukemia.py
```

This command writes `data/processed/leukemia.csv`. Do not run it against an
unverified or substituted dataset.

### Shared ID3 and corrected k-NN experiments

Both maintained runners require the processed dataset and write CSV outputs
under `results/tables/`:

```bash
PYTHONPATH=src python src/run_id3_experiment.py \
  --output-stem shared_3x5_id3_reproduction

PYTHONPATH=src python src/compare_leakage.py \
  --output-stem shared_3x5_knn_reproduction
```

For bounded smoke runs, both runners support `--max-folds N`. The defaults
use all 15 manifest folds. These are computational experiments and may
overwrite a chosen output stem; use a new stem and preserve existing
artifacts.

### Phase 7 analysis reproduction

The maintained analysis generator reads saved ID3 and corrected Phase 6 k-NN
outputs. It does not require the dataset when those result inputs are already
present.

To generate into staging locations without replacing repository artifacts:

```bash
mkdir -p /tmp/phase7_tables /tmp/phase7_figures
PYTHONPATH=src python src/analyze_phase7_results.py \
  --output-dir /tmp/phase7_tables \
  --figure-dir /tmp/phase7_figures \
  --manifest-output /tmp/phase7_reproducibility_manifest.json
```

The exact historical regeneration command recorded in the Phase 9B manifest
was:

```bash
PYTHONPATH=src python src/analyze_phase7_results.py \
  --overwrite \
  --manifest-output reports/phase9b_reproducibility_manifest.json
```

The `--overwrite` form must only be used after backing up the existing
tables, figures, and manifest.

### Figure regeneration

```bash
PYTHONPATH=src python src/plot_research_results.py \
  --output-dir /tmp/research_figures
```

To replace the repository's current figure files intentionally:

```bash
PYTHONPATH=src python src/plot_research_results.py \
  --overwrite
```

This plotting command reads existing CSVs only and does not require the
restricted dataset.

### Legacy and alternative experiments

`src/run_nested_experiment.py`, `src/run_safe_experiment.py`, and
`src/measure_feature_stability.py` contain top-level execution code rather
than the maintained CLI pattern used by the shared runners. They require the
processed dataset and write fixed historical output paths. **RECOMMENDATION:**
do not execute them without a backup and an explicit decision to reproduce
their separate protocols.

## H. Results and provenance

### Current artifacts

**VERIFIED:** The repository contains:

- Shared ID3 and corrected k-NN fold-level results, summaries, predictions,
  selected-feature records, and confusion matrices.
- Phase 7 paired-difference tables and summaries.
- Phase 5/7 historical figures.
- Current validated figures:
  `reports/figures/leakage_comparison.png`,
  `reports/figures/model_comparison.png`, and
  `reports/figures/feature_stability.png`.
- `reports/phase9b_reproducibility_manifest.json`.

The plotting script validates the schema and arithmetic of its three source
tables before creating figures. Its focused tests verify known summary values,
missing-value rejection, and dataset-independent figure generation.

### Numerical reproducibility

**VERIFIED:** `reports/phase9b_reproducibility_report.md` records that all
six Phase 7 CSV tables reproduced byte-for-byte from the maintained analysis
workflow. The three original Phase 7 PNGs were reconstructed numerically but
were not byte-identical because the original plotting code/rendering metadata
were not preserved.

The current plotting script is a later maintained visualization workflow. Its
three generated figures are reproducible from the current CSV inputs, but
pixel identity with historical figures is not established.

### Independent scientific replication

**NOT ESTABLISHED FROM THE CURRENT REPOSITORY:** Exact scientific replication
would require an authorized copy of the exact matrix, the missing metadata and
provenance decisions, and ideally an independent cohort. Passing code tests
verifies implementation behavior and data transformations; it does not
establish independent biological or clinical replication.

### Historical result conflicts

The repository preserves historical diagnostics and reports. For example,
the methodology audit documents a discrepancy between the stale ID3 MI
diagnostic summary and the standalone/main implementation. The report
concluded that the stale diagnostic was associated with a duplicate/backup
implementation and that it must not be treated as interchangeable evidence.

Older reports also predate the maintained Phase 7 and plotting scripts. For
example, `phase8_scientific_audit.md` and older checklist rows say that no
dedicated Phase 7 generator or final figures existed. Those statements are
historical snapshots, not descriptions of the current `main` tree. The
current files and commit history establish that those capabilities were added
later.

## I. Automated quality checks

| Check | Current evidence | Status |
|---|---|---|
| Python compilation | Local compilation of substantive `src/*.py` and `tests/*.py` | **VERIFIED: passed** |
| Automated tests | `python -m unittest discover -s tests -p 'test_*.py'` | **VERIFIED: 21 passed** |
| Dependency consistency | `python -m pip check` | **VERIFIED: passed** |
| GitHub Actions | Run #6 for commit `50e17a9` | **VERIFIED: success** |
| Data-independent validation | Manifest structure, plotting transformations, source analysis validation | **VERIFIED** |
| Dataset-dependent validation | Full experiments and class-label manifest validation require local data | **CONDITIONAL** |
| Full experiment rerun for this report | Not performed | **NOT PERFORMED** |

The workflow compiles `src` and `tests`, but macOS AppleDouble files named
`._*` exist in the local filesystem and are ignored. Those metadata files are
not substantive source and are not tracked.

## J. Known limitations and threats to validity

The following limitations are supported by `dataset_audit.md`,
`phase8_scientific_audit.md`, `phase9a_dataset_provenance.md`,
`phase9b_reproducibility_report.md`, and `phase10_publication_readiness.md`.

1. **Unresolved data-use terms.** The Stanford mirror has no documented
   license or redistribution permission in the repository. This blocks
   unrestricted data redistribution and may affect venue compliance.
2. **Incomplete upstream history.** The Stanford page and provenance report
   indicate that the large-matrix transformation history is lost. Numeric
   preprocessing before the local preparation script cannot be reconstructed.
3. **Synthetic feature names.** `gene_####` names are positional labels, not
   verified genes or probes. Biological, pathway, biomarker, and feature-level
   interpretation is unsupported.
4. **Missing sample metadata.** Patient IDs, batch, site, relatedness, and
   duplicate-sample information are unavailable. Patient independence and
   technical target leakage cannot be assessed.
5. **Small, high-dimensional sample.** The data contain 72 observations and
   7,128 features, with a 47/25 class split. Estimates are protocol-local and
   potentially noisy.
6. **Repeated-CV dependence.** Patients recur in test folds and training sets
   overlap. The 15 fold rows are not independent replications.
7. **Descriptive uncertainty only.** The 20,000-resample percentile bootstrap
   intervals are labeled descriptive sensitivity summaries, not inferential
   confidence intervals or significance tests.
8. **No external validation cohort.** The current evidence cannot support
   generalization or clinical utility beyond this dataset and protocol.
9. **Separate stability protocol.** `feature_stability.csv` selects 100
   features over 20 stratified training-subset resamples, whereas the shared
   performance comparison selects 50 features over 15 matched outer folds.
   Their Jaccard values must not be pooled.
10. **Legacy execution surfaces.** Several historical scripts execute at
    import/top level and write fixed outputs. They are less safe to rerun
    than the maintained shared runners and analysis scripts.

## K. Reproduction checklist

- [ ] Confirm authorized access and intended use of the exact Stanford
      `leukemia_big.csv` mirror.
- [ ] Do not add raw or processed data to Git or GitHub.
- [ ] Record the raw-file SHA-256 and compare it with
      `phase9a_dataset_provenance.md`.
- [ ] Create a CPython 3.12 environment and install
      `requirements.txt`.
- [ ] Set `PYTHONPATH=src`.
- [ ] Run `pip check`, Python compilation, and the 21-test suite.
- [ ] Inspect `configs/shared_experiment_config.json`.
- [ ] Validate `configs/outer_fold_manifest.csv` against the authorized
      processed labels locally.
- [ ] Run `src/prepare_leukemia.py` only after source and permissions are
      verified.
- [ ] Use new output stems for any ID3 or corrected k-NN experiment rerun.
- [ ] Preserve the original tables and figures before using `--overwrite`.
- [ ] Run `src/analyze_phase7_results.py` into a staging directory first.
- [ ] Compare regenerated CSV bytes and numerical summaries with the saved
      artifacts.
- [ ] Run `src/plot_research_results.py` into a staging figure directory.
- [ ] Record Python, package, operating-system, command, input hashes, and
      output hashes.
- [ ] Document any deviations, failed checks, or platform differences.
- [ ] Keep historical protocols separate from the shared 3-fold × 5-repeat
      comparison.

## L. Current readiness assessment

| Activity | Assessment | Basis |
|---|---|---|
| Code sharing | **READY** | Public repository, documented structure, pinned direct dependencies, CI workflow, tests, and no restricted dataset in the repository |
| Reproduction by an authorized researcher | **CONDITIONALLY READY** | Analysis and figures can run from saved CSVs without data; full experiments require authorized data and unresolved provenance/use decisions |
| Independent scientific replication | **NOT ESTABLISHED** | One dataset, missing sample/provenance metadata, no independent cohort, and no independent rerun documented |
| Conference manuscript submission | **NOT READY for unrestricted submission** | Phase 10 identifies unresolved data-use terms, unsupported biological claims, dependence limitations, and no external validation |

The passing test suite should be interpreted as evidence about software
behavior, not as proof that the scientific conclusions are valid. Manuscript
preparation would require, at minimum, an explicit data-use position,
restricted methodological claims, transparent provenance limitations, and a
venue-specific assessment of data and artifact requirements.

