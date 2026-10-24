# Phase 10: Publication Readiness and Dataset Decision

## Executive verdict

**Current decision: NO-GO for unrestricted manuscript drafting or submission.**

The project is **conditionally suitable for a narrow methodological
demonstration**, but only after the data-use position is clarified and the
paper scope is explicitly restricted. The current evidence can support a
descriptive statement about how deliberately using held-out labels during
feature selection changed observed scores in this saved 72-sample,
7,128-feature protocol. It cannot support biological interpretation, clinical
utility, universal classifier superiority, or generalization beyond this
dataset and protocol.

The primary publication blocker is not a missing experiment. It is the
unresolved permission and provenance status of the exact transformed matrix.
The secondary blockers are absent sample/batch metadata, synthetic feature
identifiers, dependence among repeated-CV folds, and lack of an external
validation cohort.

This report does not authorize manuscript writing, a new experiment, dataset
replacement, or modification of existing results.

## Evidence already established

### Dataset identity and provenance

**VERIFIED**

- The local raw file is byte-for-byte identical to the Stanford CASI
  `leukemia_big.csv` mirror. Its recorded SHA-256 is
  `35e84928da625da0787efb31a451dedbdf390e821a94ef74b7b7ab6cab9466d4`.
- The Stanford page reports 72 leukemia samples, 47 ALL, 25 AML, and 7,128
  features. These dimensions and labels match the local files.
- The associated Golub et al. (1999) publication citation is supported by the
  Stanford page and Europe PMC.
- The local preparation code only transposes the matrix, generates
  `gene_0001` through `gene_7128`, appends the target, and validates shape,
  labels, missingness, and finiteness.

**UNRESOLVED**

- The original accession or source file for the exact transformed Stanford
  matrix has not been verified.
- The Stanford page states that the large-matrix transformation was lost.
- The original feature identifiers, sample identities, upstream preprocessing,
  batch information, and independent-patient status are unavailable locally.
- No explicit license, research-use statement, or redistribution permission
  was found for the exact Stanford mirror.
- OpenML dataset 1104 is a related representation, not an exact match to the
  local file; its `Public` metadata must not be treated as a license for the
  local matrix.

Evidence and external links are documented in
[`phase9a_dataset_provenance.md`](./phase9a_dataset_provenance.md).

### Experimental design and leakage contrast

**VERIFIED**

- ID3 and corrected k-NN use the same 15-fold manifest: stratified 3-fold
  cross-validation repeated five times, with 48 training and 24 test samples
  per fold.
- Both models use 50 selected features and the same safe/leaky outer splits.
- In the safe condition, feature selection receives training observations and
  training labels only.
- In the deliberately leaky condition, all labels, including held-out
  labels, are supplied only to the feature-selection step.
- Training-fitted imputation and scaling are applied in the corrected k-NN
  pipeline; the classifier is fit on training observations only.
- ID3 discretization is fitted on training-selected features only, and the
  tree is fit on training observations only.
- The maintained analysis script validates the fold contract and reads the
  corrected Phase 6 k-NN output rather than the superseded k-NN results.

The relevant implementation is
[`analyze_phase7_results.py`](../src/analyze_phase7_results.py), together
with [`run_id3_experiment.py`](../src/run_id3_experiment.py) and
[`compare_leakage.py`](../src/compare_leakage.py).

### Saved results and reproducibility

**VERIFIED**

- Phase 7 contains eight paired leakage summaries: two classifiers, two
  selectors, and two metrics.
- The observed leaky-minus-safe mean differences are positive in all eight
  comparisons:
  - ID3 + ANOVA: Macro-F1 `+0.009097`; balanced accuracy `+0.008333`.
  - ID3 + MI: Macro-F1 `+0.028769`; balanced accuracy `+0.027315`.
  - k-NN + ANOVA: Macro-F1 `+0.005878`; balanced accuracy `+0.003843`.
  - k-NN + MI: Macro-F1 `+0.021462`; balanced accuracy `+0.021528`.
- The saved analysis uses descriptive percentile bootstrap intervals over
  paired fold rows, not significance tests.
- The Phase 9B manifest records the command, Python/package versions, input
  hashes, output hashes, and bootstrap settings. An independent check found
  all 14 recorded input/output hashes matched the current files.
- All six Phase 7 CSV outputs were reproduced byte-for-byte by the maintained
  analysis pipeline. The three figures were reconstructed numerically but
  are not byte-identical to the historical PNGs because the original plotting
  session was not preserved.
- The full test suite was recorded as 18 passing tests in Phase 9B.

These results are documented in
[`phase7_final_comparative_analysis.md`](./phase7_final_comparative_analysis.md)
and [`phase9b_reproducibility_report.md`](./phase9b_reproducibility_report.md).

### Statistical scope

**VERIFIED**

- The 15 rows per comparison are repeated-CV fold results, not independent
  experimental replications.
- Patients recur in test sets across repeats, and training sets overlap across
  folds and repeats.
- The existing reports explicitly describe the bootstrap intervals as
  descriptive sensitivity summaries.

**RECOMMENDATION**

For the current dataset, keep the principal analysis descriptive. Do not add
p-values or label the current intervals as inferential confidence intervals.
If the paper seeks inferential claims, pre-specify a dependence-aware method
and justify its resampling unit or correlation structure before analysis.
An independent validation cohort would address a more consequential limitation
than applying a different test to the same 72 observations.

## Outstanding blockers

| Finding | Status | Publication consequence |
|---|---|---|
| Exact-data license and permitted redistribution/use are unknown | UNRESOLVED | Raw-data sharing and some venue data-availability requirements cannot yet be satisfied |
| Upstream transformation history is lost | VERIFIED / UNRESOLVED | The matrix can be identified as the Stanford mirror, but its full provenance cannot be reconstructed |
| Original feature identifiers are unavailable | VERIFIED / UNRESOLVED | No gene-level, pathway, biomarker, or biological feature claims are supportable |
| Sample IDs, batch, site, relatedness, and duplicate-sample metadata are unavailable | VERIFIED / UNRESOLVED | Patient independence and technical confounding cannot be fully assessed |
| No independent external validation cohort exists | VERIFIED | Generalization and clinical utility claims are not supportable |
| Repeated-CV rows are dependent | VERIFIED | Current uncertainty summaries must remain descriptive |
| Target venue and its data/code policies are not selected | UNRESOLVED | Compliance cannot be assessed against a specific conference or journal |

## Dataset decision options

### Option A: retain the Stanford-hosted dataset and seek clarification

**Scientific advantages**

- Preserves the exact data used for all existing experiments and avoids
  changing the evidence base.
- The immediate public mirror and Golub citation are already documented.
- The current results remain directly reproducible from the saved hashes and
  manifest if the data-use position is clarified.

**Risks and limitations**

- The exact upstream accession, transformation history, sample metadata, and
  feature mapping may remain unavailable even after contacting the source.
- Public downloadability does not establish permission to redistribute the
  matrix or derived data.
- Without feature mapping, the paper remains methodological rather than
  biological.

**Effect on existing experiments**

- No effect if the file is retained unchanged. Existing results, hashes,
  protocols, and reports remain valid for the current matrix.
- A newly discovered source correction or data replacement would require an
  explicit decision and would not silently invalidate or overwrite historical
  results.

**Additional work required**

1. Contact the Stanford/CASI maintainer or another authoritative data steward.
2. Request the applicable research-use and redistribution terms for the exact
   file, its provenance, and any available transformation or feature mapping.
3. Preserve the response and cite it in the data-availability documentation.
4. Confirm whether publishing derived tables, selected positional features, and
   code is permitted even if raw-data redistribution is not.

**Support for intended claims**

- Supports a narrow, protocol-local methodological paper if terms permit the
  intended use and the paper discloses all unresolved metadata.
- Does not by itself support biological interpretation, clinical claims, or
  universal conclusions.

**Assessment: preferred first action.** It is the least disruptive path and
  may resolve the most important blocker without rerunning experiments.

### Option B: find an independently documented dataset

**Scientific advantages**

- A dataset with a verified accession, feature identifiers, sample metadata,
  and explicit terms would improve provenance and biological interpretability.
- A second independent dataset could test whether the observed leakage pattern
  transfers beyond this one matrix.
- External validation would strengthen generalization claims more directly than
  further resampling of the current folds.

**Risks and limitations**

- A different dataset may differ in platform, class balance, preprocessing,
  feature space, and clinical question.
- The current Stanford results would remain a historical single-dataset
  analysis; they could not be pooled with the new dataset as if they were
  paired observations.
- Dataset selection can introduce a new source of researcher degrees of
  freedom unless inclusion criteria and analysis decisions are documented
  before rerunning.

**Effect on existing experiments**

- Existing results must be preserved and labeled as the original-dataset
  analysis.
- The new dataset requires a separate, explicitly documented experiment and
  likely a new shared-fold manifest or an external-validation design.
- No current result should be relabeled as a result from the new dataset.

**Additional work required**

1. Define eligibility criteria: accession, feature identity, labels,
   metadata, license, and compatibility with the leakage question.
2. Verify values, orientation, sample counts, class labels, feature order, and
   preprocessing history.
3. Pre-register or document the new protocol and rerun the relevant
   safe/leaky comparisons.
4. Report the original and independent datasets separately, then assess
   replication descriptively.

**Support for intended claims**

- Best option for claims about transferability or robustness across datasets.
- Still does not justify universal classifier or selector claims from only one
  additional dataset.

**Assessment: scientifically strongest for broader claims, but it is new
work and is not a prerequisite for a narrowly scoped methods demonstration if
Option A or C is accepted.**

### Option C: retain the current dataset as a limited methodological demonstration

**Scientific advantages**

- Uses the completed, internally verified experiments without rerunning them.
- Directly demonstrates the distinction between training-only feature
  selection and deliberate held-out-label contamination.
- The result is reproducible at the table level and can be framed as a
  controlled computational example.

**Risks and limitations**

- Provenance, licensing, transformation history, sample independence, and
  biological feature identity remain unresolved.
- The results cannot establish that the observed leakage magnitude is typical
  of genomic studies or other datasets.
- Synthetic positional names prevent biological interpretation.
- Venue reviewers may reject the data-availability or provenance position even
  with full disclosure.

**Effect on existing experiments**

- No effect; existing results can be retained unchanged.
- The paper must not silently imply that `gene_####` names are verified genes
  or that the leaky condition is a valid deployment estimate.

**Additional work required**

1. Obtain a documented data-use decision from the target venue or source
   steward before submission.
2. Cite the Stanford mirror and Golub paper, record the exact file hash, and
   disclose the lost transformation and missing mapping.
3. Treat the features as anonymous positional predictors.
4. Keep all claims descriptive and protocol-local.
5. Provide code, configuration, manifest, result hashes, and a clear
   non-redistribution/data-access statement where applicable.

**Support for intended claims**

- Supports only the restricted question: what changed in this saved
  72-sample protocol when held-out labels were allowed into selection?
- Does not support biological, clinical, external-validity, or universal
  superiority claims.

**Assessment: viable fallback only with explicit provenance/licensing
disclosure and venue acceptance.**

## Recommended research question and scope

### Defensible question

**RECOMMENDATION**

Use a protocol-qualified version of the existing question:

> Within a 72-sample, 7,128-feature leukemia expression matrix and a fixed
> repeated stratified cross-validation protocol, how do observed Macro-F1 and
> balanced-accuracy scores change when held-out labels are deliberately made
> available to ANOVA or mutual-information feature selection, and how does
> that descriptive pattern differ between the fixed ID3 and k-NN
> implementations?

This wording makes the information boundary and the dataset limitation
explicit. It does not call the leaky condition a valid model, and it avoids
claiming a universal effect.

### What the current experiments support

**VERIFIED**

- In the saved protocol, all eight mean leaky-minus-safe differences are
  positive.
- The mean increase is larger for mutual information than ANOVA for both
  classifiers in the current folds.
- Safe k-NN scores are descriptively higher than safe ID3 scores on most
  matched folds, but this is a one-dataset comparison.
- The repeated-CV analysis is reproducible at the table level and does not
  require treating the 15 folds as independent replications.

### What requires additional datasets

**RECOMMENDATION**

Additional datasets are required for:

- claims that the leakage effect is typical across genomic datasets;
- claims of robustness across platforms, cohorts, or preprocessing pipelines;
- claims about external validity or clinical utility;
- claims that one classifier or selector is generally superior.

### What requires stronger statistical analysis

**RECOMMENDATION**

A dependence-aware inferential analysis would be required for claims such as
“the leakage effect is statistically significant,” “MI reliably causes more
inflation than ANOVA,” or “k-NN outperforms ID3.” No such claims should be
made from the current descriptive intervals.

### What requires verified feature identifiers

**RECOMMENDATION**

Verified identifiers and provenance are required for:

- naming selected genes or probes;
- pathway or functional enrichment;
- biomarker or disease-mechanism claims;
- biological explanations for selector or classifier differences.

None of these interpretations is justified by the current positional names.

## Statistical plan decision

**RECOMMENDATION: keep the current paper descriptive unless the scope is
expanded.**

For this 72-observation, 7,128-feature dataset, the current paired fold
summaries and descriptive bootstrap sensitivity intervals are adequate for a
transparent methodological demonstration. They are not adequate for
population-level inference. A dependence-aware inferential analysis is
essential only if the paper seeks inferential or general claims; it is not a
reason to manufacture p-values for the current narrow demonstration.

If stronger inference is later required, the method must be selected and
documented before rerunning. It should account for repeated use of patients,
overlapping training sets, the paired selector/classifier structure, and the
fact that the deliberate leakage control is not a valid prediction procedure.
Subject-level resampling, a prespecified repeated-CV dependence correction, or
an external-validation analysis may be considered, but none is implemented or
endorsed as the answer in this phase.

## Publication questions before choosing a venue

These are decision questions, not guarantees about any particular venue:

1. **Data-use terms:** Does the venue permit publication using a public mirror
   whose exact license and redistribution terms are unresolved? May the paper
   publish derived tables and selected positional features without
   redistributing the raw matrix?
2. **Data citation:** What source citation, accession, persistent URL, hash,
   and data-availability statement does the venue require when no verified
   accession exists?
3. **Reproducibility:** Must reviewers receive the raw data, or is code,
   configuration, manifest, hashes, and instructions for authorized access
   sufficient?
4. **Human-data governance:** Does the venue require an ethics, consent,
   de-identification, or data-governance statement for a leukemia patient
   expression dataset whose sample metadata are unavailable?
5. **Novelty:** Is a controlled demonstration of feature-selection leakage
   with fixed ID3 and k-NN settings sufficiently novel for the intended
   audience, or is an external replication expected?
6. **External validation:** Does the venue expect an independent cohort,
   cross-platform validation, or only a methodological proof of concept?
7. **Biological claims:** Are synthetic feature names acceptable for the
   venue's scope, and can the paper remain entirely methodological if no
   mapping is recovered?
8. **Artifact policy:** Are the maintained analysis script, test suite,
   reproducibility manifest, and non-redistributable-data instructions
   sufficient for artifact evaluation?

No target venue has yet been selected, so these questions cannot be resolved
from the project files alone. Acceptance should not be predicted from this
audit.

## Minimum additional work, prioritized

### P0: required before submission

1. Resolve the exact-data use/redistribution position with the Stanford/CASI
   maintainer or another authoritative steward, or obtain a documented
   venue/editor decision that the disclosed use is acceptable.
2. Preserve the response, source URL, raw-file hash, and access date in the
   project provenance record.
3. Decide explicitly between Option A and the restricted form of Option C.
4. Freeze the claim scope: descriptive, protocol-local, no biological
   interpretation, no clinical utility, and no universal superiority claims.

### P1: required for a credible methods paper

1. Prepare a data-availability and reproducibility package using the Phase 9B
   script, manifest, tests, configuration, fold manifest, and output hashes.
2. Document that `gene_####` values are positional synthetic names and that
   the upstream transformation and feature mapping are unavailable.
3. Document the dependence of repeated-CV folds and keep bootstrap language
   descriptive.
4. Confirm the target venue's requirements for non-redistributable or
   incompletely licensed data and human-data governance.

### P2: required for broader scientific claims

1. Add an independently documented, licensed dataset or external validation
   cohort under a separately specified protocol.
2. Verify feature identifiers and relevant sample/batch metadata if biological
   interpretation is desired.
3. Design any dependence-aware inferential analysis before collecting new
   results.

## Conditions under which manuscript drafting can begin

Drafting may begin only after all of the following are true:

1. The data-use position for the exact Stanford mirror is documented, or the
   target venue explicitly accepts the stated limitation.
2. The manuscript scope is restricted to a methodological demonstration on the
   observed dataset and protocol.
3. The paper explicitly states that the leaky condition is a deliberate
   diagnostic control, not a valid deployment pipeline.
4. The paper does not name synthetic positional columns as genes, make
   biological interpretations, claim clinical utility, claim statistical
   significance, or claim universal classifier/selector superiority.
5. The data-availability, code, configuration, manifest, test, hash, and
   figure-reproduction plan is accepted for the target venue.
6. Any required ethics, consent, de-identification, or human-data governance
   statement has been resolved.

Until these conditions are met, manuscript preparation is **NO-GO**. No
experiment rerun or result modification is required to make this decision.

## Audit limitations

This review used the project reports, source files, saved result tables,
reproducibility manifest, and focused hash/summary checks. It did not contact
the data steward, inspect private metadata, select a venue, obtain legal
advice, or run new experiments. A venue editor, data steward, institutional
review process, or legal authority may reach a different decision based on
information unavailable in this workspace.
