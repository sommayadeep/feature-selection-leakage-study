# Completion Checklist

Status meanings:

- **COMPLETE**: verified by inspection or execution for the stated scope.
- **PARTIAL**: some evidence or implementation exists, but validation or
  completion remains.
- **BLOCKED**: progress requires missing data, credentials, external access, or
  an unresolved decision.
- **NEEDS HUMAN REVIEW**: requires a scientific, ethical, provenance, or
  submission decision.

This checklist includes the completed Phase 4 controlled experiments. No
historical result table was modified.

## Phase 1: project inspection

| Task | Status | Evidence or remaining work |
|---|---|---|
| Inspect complete local project structure | COMPLETE | Source, data, output, environment, and empty documentation/output directories were inventoried |
| Inspect raw and processed datasets | COMPLETE | See [`dataset_audit.md`](./dataset_audit.md) |
| Inspect `requirements.txt` and existing environment | COMPLETE | `.venv` is Python 3.12.13; requirements contains 113 exact pins |
| Inspect experiment scripts | COMPLETE | ID3, k-NN, nested-CV, stability, preparation, and analysis scripts read |
| Inspect diagnostic scripts | COMPLETE | Conflicting old and standalone MI diagnostics identified |
| Inspect existing CSV result tables | COMPLETE | All 15 expected substantive tables exist and schemas/row counts were checked |
| Inspect figures, logs, notebooks, and manuscript locations | COMPLETE | These directories are empty |
| Create project inventory | COMPLETE | [`project_inventory.md`](./project_inventory.md) |
| Create methodology audit | COMPLETE | [`methodology_audit.md`](./methodology_audit.md) |
| Create dataset audit | COMPLETE | [`dataset_audit.md`](./dataset_audit.md) |
| Create Phase 1 completion checklist | COMPLETE | This file |

## Phase 2: preserve and verify existing results

| Task | Status | Evidence or remaining work |
|---|---|---|
| Preserve original datasets and result tables | COMPLETE | Phase 2 backup created at `reports/phase2_backup_20261024_0913/`; result tables remain byte-for-byte unchanged |
| Back up existing scripts before source changes | COMPLETE | Original substantive Python scripts copied to the Phase 2 backup directory |
| Record original result checksums | COMPLETE | `original_source_and_result_checksums.sha256` records SHA-256 values; all original result-table checksums still match |
| Recompute ID3 aggregate values from saved fold rows | COMPLETE | Saved means match the reported ID3 values at the displayed precision |
| Recompute preliminary k-NN aggregate values from saved table | COMPLETE | Saved `leakage_comparison.csv` matches the reported preliminary values |
| Independently rerun all existing experiments | PARTIAL | Not performed in Phase 1 to avoid expensive runs; one standalone MI fold reproduces its saved row |
| Resolve conflicting MI diagnostic outputs | COMPLETE | In-memory comparison showed the stale diagnostic scores exactly match the duplicate backup tree implementation; the main/standalone implementation matches the saved fold and feature records |
| Verify every reported value for final reporting | PARTIAL | CSV-internal checks pass for the main summaries; full independent verification remains |

## Phase 2: implementation and controlled verification

| Task | Status | Evidence or remaining work |
|---|---|---|
| Make `run_id3_experiment.py` import-safe | COMPLETE | Executable work is inside `main()` and an `if __name__ == "__main__"` guard; import test passed without changing result hashes |
| Preserve ID3 algorithm and defaults | COMPLETE | Tree, selector, CV, and output logic were retained; only execution placement changed |
| Add entropy/information-gain tests | COMPLETE | Synthetic `unittest` cases pass |
| Add tree edge-case tests | COMPLETE | Pure nodes, exhausted features, empty labels, depth, minimum samples, and deterministic behavior pass |
| Add leakage-boundary tests | COMPLETE | Synthetic safe-selector and matched-split tests pass |
| Propose shared experiment configuration | COMPLETE | [`src/experiment_config.py`](../src/experiment_config.py) and [`configs/shared_experiment_config.json`](../configs/shared_experiment_config.json) |
| Generate shared outer-fold manifest | COMPLETE | [`configs/outer_fold_manifest.csv`](../configs/outer_fold_manifest.csv) contains 15 deterministic 3x5 folds and matches the legacy ID3 splitter |
| Integrate the new manifest into comparable runners | COMPLETE | `run_id3_experiment.py`, `run_safe_experiment.py`, and `compare_leakage.py` consume the validated 15-split manifest; nested-CV and stability designs remain separate |
| Document Phase 2 changes | COMPLETE | [`phase2_implementation_report.md`](./phase2_implementation_report.md) |

## Phase 3: research questions

| Task | Status | Evidence or remaining work |
|---|---|---|
| Record initial RQ1-RQ4 framework | COMPLETE | Defined in the master project instruction |
| Test whether current designs answer each RQ | PARTIAL | Current outputs cover preliminary ID3 and k-NN comparisons but not all stability and uncertainty requirements |
| Refine research questions based on validated evidence | PARTIAL | Defer until the protocol and diagnostic conflict are resolved |

## Phase 4: dataset audit

| Task | Status | Evidence or remaining work |
|---|---|---|
| Establish sample/feature counts and classes | COMPLETE | 72 samples, 7,128 features, 47 ALL, 25 AML |
| Check missingness and finite values | COMPLETE | No missing values; all features finite |
| Check duplicate rows and columns | COMPLETE | None detected in the local processed file |
| Check constant/near-constant features | COMPLETE | None with one or two unique values |
| Check identifier/target columns | PARTIAL | No explicit identifier column; provenance and batch/group metadata are absent |
| Determine whether names are biological identifiers | COMPLETE | Names are synthetic positional names; no mapping is present |
| Document data provenance and license | BLOCKED | Source/provenance metadata is absent from the workspace |

## Phase 5: leakage-safe pipeline

| Task | Status | Evidence or remaining work |
|---|---|---|
| Shared safe k-NN pipeline keeps fit operations inside folds | COMPLETE | The shared runner uses a pipeline with fold-local imputation, scaling, selection, and fixed `k=3` |
| Shared safe ID3 path fits imputation/selection/discretization on training data | COMPLETE | The import-safe runner consumes the shared manifest and fits all transformations on each training fold |
| Verify custom ID3 edge cases | COMPLETE | Embedded toy test and 11 lightweight unit tests pass |
| Add reusable tested implementation | COMPLETE | Shared manifest and bounded runner controls are implemented and exercised |

## Phase 6: intentionally leaky comparison

| Task | Status | Evidence or remaining work |
|---|---|---|
| Existing ID3 leaky condition is explicitly labeled | COMPLETE | Code and output columns distinguish `safe` and `leaky` |
| Shared k-NN leaky condition uses held-out labels | COMPLETE | The per-fold selector is fitted on all labels, including the held-out labels, using the same outer split |
| Isolate selector leakage from preprocessing leakage | PARTIAL | The saved shared k-NN leaky path fits its first imputer/scaler on training data and uses held-out labels only for selection, but then applies an additional post-selection scaler absent from the safe path; paired k-NN analysis is blocked until corrected and rerun |
| Save per-fold leaky/safe k-NN records | COMPLETE | Shared Phase 4 runner saves fold metrics, predictions, selected features, and confusion matrices |

## Phase 7: standardized experiments

| Task | Status | Evidence or remaining work |
|---|---|---|
| Use stratified CV and fixed seeds | COMPLETE | Existing scripts use fixed seeds and stratified CV |
| Use the same outer splits across matched conditions | COMPLETE | Existing ID3 and preliminary k-NN comparisons match safe/leaky conditions within each script |
| Use one common split family across matched ID3 and future fixed k-NN | COMPLETE | Both shared runners load `configs/outer_fold_manifest.csv`; historical 3x2 and nested 3x3 outputs remain incomparable |
| Nested model selection for k-NN | PARTIAL | Implemented in a separate preliminary script, not integrated with the comparison |

## Current Phase 5: results analysis and verification

| Task | Status | Evidence or remaining work |
|---|---|---|
| Independently audit final source and shared artifacts | COMPLETE | Final row counts, duplicate keys, prediction metrics, confusion counts, fold IDs, selected-feature counts, and ID3 overlap were independently checked |
| Verify ID3 leakage boundaries | COMPLETE | Training-only imputation, safe selection, deliberately leaky label use, training-only discretization, and training-only classifier fitting were confirmed |
| Verify k-NN leakage boundaries and condition equivalence | PARTIAL | No held-out labels enter preprocessing, but the leaky path has an extra post-selection scaler; this blocks valid k-NN paired/model analysis |
| Compute verified paired differences | PARTIAL | ID3 paired table and descriptive intervals created; k-NN paired analysis is blocked pending correction and rerun |
| Compare ID3 and k-NN on shared splits | BLOCKED | `phase5_model_comparison.csv` records the blocked k-NN scope; no cross-classifier conclusion is made |
| Review feature-selection stability | COMPLETE | Historical 20-resample stability and matched 15-fold ID3 overlap are reported separately and not pooled |
| Create Phase 5 tables and figures | COMPLETE | Paired-difference table, blocked-status model-comparison table, and two protocol-labeled figures created |
| Create Phase 5 results report | COMPLETE | [`phase5_results_analysis.md`](./phase5_results_analysis.md) documents verified facts, numerical findings, caveats, and the unresolved k-NN correction |
| Document one final protocol | PARTIAL | No experimental protocol document exists |

## Phase 6: corrected k-NN preprocessing

| Task | Status | Evidence or remaining work |
|---|---|---|
| Inspect the k-NN preprocessing and shared configuration | COMPLETE | `compare_leakage.py`, `experiment_config.py`, shared JSON, manifest, and Phase 5 report were inspected |
| Correct the duplicate leaky-path scaling | COMPLETE | Shared `preprocess_fold()` now fits one training-only imputer/scaler; both conditions use the same plain k-NN classifier |
| Add regression tests for leakage boundaries | COMPLETE | 14 tests pass, including selector-label spies, training-only preprocessing, classifier row boundaries, and shared settings |
| Run corrected two-fold smoke test | COMPLETE | `smoke_knn_phase6_*` artifacts validated: 8 rows, 192 predictions, 8 selections, 8 confusion records |
| Run corrected shared 3x5 k-NN experiment | COMPLETE | `shared_3x5_knn_phase6_corrected_*` artifacts contain all 15 folds and four conditions |
| Validate corrected outputs against predictions and manifest | COMPLETE | Metrics, confusion counts, summaries, fold metadata, counts, and feature cardinalities independently recomputed |
| Preserve previous k-NN outputs and backups | COMPLETE | Existing `shared_3x5_knn_*` files remain separate; corrected run used a new output stem |
| Document the correction | COMPLETE | [`phase6_knn_correction_report.md`](./phase6_knn_correction_report.md) |

## Phase 7: final comparative analysis

| Task | Status | Evidence or remaining work |
|---|---|---|
| Verify ID3 and corrected k-NN fold pairing | COMPLETE | Both result tables contain the same 15 manifest folds, paired safe/leaky rows, and matching 48/24 train/test metadata |
| Recompute predictions and confusion checks | COMPLETE | 1,440 predictions and 60 confusion records per model independently reproduce saved metrics and counts |
| Calculate paired leakage differences | COMPLETE | [`phase7_paired_differences.csv`](../results/tables/phase7_paired_differences.csv) contains 120 fold-level metric rows across four combinations |
| Compare safe ID3 and corrected k-NN | COMPLETE | [`phase7_model_comparison.csv`](../results/tables/phase7_model_comparison.csv) contains matched-fold descriptive differences |
| Compare ANOVA and MI leakage differences | COMPLETE | [`phase7_selector_comparison.csv`](../results/tables/phase7_selector_comparison.csv) contains matched-fold selector contrasts |
| Review feature stability protocols separately | COMPLETE | Historical 100-feature resampling and 50-feature ID3 overlap values remain separate |
| Create publication-readable figures | COMPLETE | Three Phase 7 figures created under `reports/figures/` |
| Document Phase 7 analysis and limitations | COMPLETE | [`phase7_final_comparative_analysis.md`](./phase7_final_comparative_analysis.md) |

## Phase 3: protocol validation

| Task | Status | Evidence or remaining work |
|---|---|---|
| Validate manifest row count, stratification, disjointness, and patient repetition | COMPLETE | `load_outer_split_manifest()` validates all 15 rows; each patient appears in exactly five test folds |
| Verify identical ID3 and k-NN splits | COMPLETE | Both shared runners load the same manifest; synthetic and real-label tests compare every train/test index |
| Audit safe preprocessing boundaries | COMPLETE | Pipelines and per-fold code fit imputation, scaling, selection, and ID3 discretization on training data only |
| Separate intentional selector leakage from preprocessing leakage | PARTIAL | Held-out labels are confined to selection and the first imputer/scaler are fold-local, but Phase 5 found an extra post-selection scaler in the leaky k-NN path |
| Preserve historical results and backups | COMPLETE | New protocol uses new output filenames; original result checksums remain unchanged |
| Run lightweight validation | COMPLETE | 11 unit tests, Python compilation, and Pylance diagnostics passed |
| Produce Phase 3 protocol report | COMPLETE | [`phase3_protocol_validation.md`](./phase3_protocol_validation.md) |

## Phase 4: controlled shared experiments

| Task | Status | Evidence or remaining work |
|---|---|---|
| Complete pre-run validation | COMPLETE | Manifest use, leakage boundaries, output paths, syntax, and unit tests checked before execution |
| Run bounded smoke tests | COMPLETE | Two-fold ID3 and k-NN smoke outputs validated; smoke files are excluded from paper claims |
| Run full shared ID3 experiment | COMPLETE | 15 folds x 2 selectors x 2 conditions; 60 fold-condition rows and 1,440 predictions |
| Run full shared k-NN experiment | COMPLETE | Same 15 folds x 2 selectors x 2 conditions; 60 fold-condition rows and 1,440 predictions |
| Recompute aggregate metrics from fold rows | COMPLETE | Saved summaries agree with independently recomputed means and standard deviations |
| Verify paired outer folds across models | COMPLETE | Both result tables contain global folds 1 through 15 with no duplicates |
| Preserve historical result tables | COMPLETE | All 30 backed-up historical CSVs are byte-for-byte unchanged |
| Save final comparison artifacts | COMPLETE | Fold metrics, summaries, selected features, predictions, and confusion matrices saved for both models |
| Document Phase 4 execution and limitations | COMPLETE | [`phase4_experiment_report.md`](./phase4_experiment_report.md) |

## Phase 8: feature stability

| Task | Status | Evidence or remaining work |
|---|---|---|
| Compute pairwise Jaccard similarity | COMPLETE | `feature_stability.csv` and `id3_feature_overlap.csv` exist |
| Report Jaccard standard deviation | COMPLETE | Present in `feature_stability.csv` and aggregate ID3 analysis |
| Report selection frequencies | PARTIAL | No frequency table exists |
| Compare safe/leaky stability using matched resamples | PARTIAL | Current stability experiment is training-resample only and has no leaky condition |
| Align stability feature count with performance experiments | PARTIAL | Stability uses 100 features; ID3 comparison uses 50 |

## Phase 9: performance differences

| Task | Status | Evidence or remaining work |
|---|---|---|
| Compute paired per-fold differences for ID3 | COMPLETE | 30 matched ID3 rows exist |
| Count positive/zero/negative differences | COMPLETE | Descriptive categories exist in `id3_fold_difference_analysis.csv` |
| Report mean/median/std differences | COMPLETE | Present for ID3 |
| Add paired uncertainty/effect-size analysis | COMPLETE | Phase 5 saves descriptive paired ID3 differences and percentile bootstrap intervals with explicit repeated-CV dependence caveats; k-NN remains blocked |
| Add confusion matrices for final comparisons | COMPLETE | Final ID3 and k-NN confusion-matrix CSV artifacts were independently checked against predictions |
| Distinguish descriptive association from causation | COMPLETE | Existing analysis scripts include descriptive caveats |

## Phases 10-14

| Phase/task | Status | Evidence or remaining work |
|---|---|---|
| Robustness and second-dataset assessment | PARTIAL | No second dataset is present or validated |
| Final table/figure generator | PARTIAL | Existing partial analysis scripts exist; final generator and figures are absent |
| Reproducibility documentation | PARTIAL | Requirements and scripts exist; README/protocol/run manifest are absent |
| Lightweight tests and smoke-test command | COMPLETE | 11 lightweight tests and bounded smoke-test artifacts exist; full k-NN paired interpretation remains blocked by the preprocessing discrepancy |
| Conference manuscript | PARTIAL | `paper/` is empty |
| Final completion report | PARTIAL | This Phase 1 checklist is not the final completion report |

## Phase 8: scientific and statistical audit

| Task | Status | Evidence or remaining work |
|---|---|---|
| Audit safe/leaky preprocessing and classifier boundaries | COMPLETE | Current ID3 and corrected k-NN source inspected; training-only fitting and intentional selector leakage confirmed |
| Verify shared-fold pairing and saved metrics | COMPLETE | Both final models independently reproduce metrics/confusions from predictions on manifest folds 1–15 |
| Audit statistical claims and repeated-CV dependence | COMPLETE | Reports make descriptive-only claims; bootstrap limitations and a dependence-aware recommendation are documented |
| Audit dataset provenance and licensing | BLOCKED | No authoritative source, citation/accession, license, feature mapping, or complete preprocessing provenance is present |
| Audit generalizability | COMPLETE | Conclusions are limited to one 72-patient dataset; no external validation cohort exists |
| Preserve rerunnable Phase 7 analysis workflow | PARTIAL | Output tables and figures exist, but no dedicated Phase 7 generator script was found |
| Create Phase 8 scientific audit | COMPLETE | [`phase8_scientific_audit.md`](./phase8_scientific_audit.md) |
| Manuscript readiness decision | NOT READY | Resolve provenance/licensing and reproducibility gaps before manuscript writing; external data is needed for broader claims |

## Phase 9A: dataset provenance and licensing investigation

| Task | Status | Evidence or remaining work |
|---|---|---|
| Inspect local dataset metadata and documentation | COMPLETE | Raw/processed files, preparation script, audits, project docs, and metadata inventory inspected; no local README, accession, or license record found |
| Record raw and processed SHA-256 hashes | COMPLETE | Hashes are recorded in `dataset_audit.md` and `phase9a_dataset_provenance.md` |
| Verify immediate public source | COMPLETE | Local raw CSV is byte-for-byte identical to Stanford CASI `leukemia_big.csv` |
| Verify underlying publication citation | COMPLETE | Stanford page and Europe PMC support Golub et al. (1999), Science 286(5439):531–537 |
| Determine original accession and feature mapping | BLOCKED | No verified accession or mapping for the transformed 7,128-feature Stanford matrix |
| Assess licensing and redistribution terms | BLOCKED | Stanford mirror has no explicit license; OpenML `Public` metadata applies to a different representation and is not sufficient |
| Create Phase 9A provenance report | COMPLETE | [`phase9a_dataset_provenance.md`](./phase9a_dataset_provenance.md) |
| Publication recommendation | CONDITIONAL | Resolve exact-data terms and provenance first; otherwise restrict claims to a transparent methodological demonstration |

## Phase 9B: reproducible final analysis pipeline

| Task | Status | Evidence or remaining work |
|---|---|---|
| Recover and maintain Phase 7 analysis logic | COMPLETE | [`../src/analyze_phase7_results.py`](../src/analyze_phase7_results.py) reads the shared ID3 and corrected Phase 6 k-NN results only |
| Preserve historical Phase 7 artifacts before regeneration | COMPLETE | [`phase9b_backup_20261024_095818/`](./phase9b_backup_20261024_095818/) contains the original tables, figures, and SHA-256 hashes |
| Reproduce Phase 7 tables | COMPLETE | All six CSVs match the pre-run versions byte-for-byte; row counts are 120, 8, 60, 4, 60, and 4 |
| Reconstruct Phase 7 figures | COMPLETE | Three publication-readable figures regenerated; exact PNG bytes are not reproducible because the original plotting session was not preserved |
| Add analysis tests and input validation | COMPLETE | [`../tests/test_phase9b_analysis.py`](../tests/test_phase9b_analysis.py); Phase 9B recorded 18 tests, and the current suite passes 21 tests |
| Record reproducibility metadata | COMPLETE | [`phase9b_reproducibility_manifest.json`](./phase9b_reproducibility_manifest.json) records command, environment, hashes, and bootstrap settings |
| Create Phase 9B report | COMPLETE | [`phase9b_reproducibility_report.md`](./phase9b_reproducibility_report.md) |
| Manuscript readiness after Phase 9B | NOT READY | Exact-data licensing/provenance and external-validation limitations from Phases 8-9A remain unresolved |

## Phase 10: publication readiness and dataset decision

| Task | Status | Evidence or remaining work |
|---|---|---|
| Review Phase 9A provenance and licensing evidence | COMPLETE | [`phase10_publication_readiness.md`](./phase10_publication_readiness.md); exact Stanford mirror identity is verified but exact-data terms remain unresolved |
| Review Phase 9B reproducibility evidence | COMPLETE | Manifest hashes, source pipeline, saved summaries, and Phase 9B report were checked; all 14 recorded hashes match |
| Verify current research-question scope | COMPLETE | Existing experiments support only a descriptive, protocol-local leakage demonstration |
| Evaluate dataset options A, B, and C | COMPLETE | Options and their effects on existing results are compared in [`phase10_publication_readiness.md`](./phase10_publication_readiness.md) |
| Decide statistical scope | COMPLETE | Retain descriptive analysis unless broader claims require a prespecified dependence-aware method |
| Identify venue and publication questions | COMPLETE | Data-use, citation, reproducibility, human-data, novelty, external-validation, feature-identity, and artifact-policy questions are listed |
| Publication readiness verdict | NOT READY | No-go for unrestricted manuscript drafting; conditional go only after data-use/venue clarification and a restricted methodological scope |
| New experiments or dataset changes | NOT PERFORMED | No experiments rerun, dataset modified, or historical result overwritten |
