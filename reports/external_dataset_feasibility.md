# External leukemia dataset feasibility assessment

**Assessment date:** 2026-10-31

**Decision:** **CONDITIONAL GO** to prepare an independent replication on MILE
Stage II (`GSE13164`), subject to data-use review and a sample/feature metadata
preflight. This is a feasibility decision only: no data were downloaded, no
experiments were run, and no existing results or protocols were changed.

## Purpose and scope

The repository's existing evidence is based on one Stanford CASI mirror of the
Golub leukemia data: 72 samples, 7,128 numeric features, 47 ALL, and 25 AML.
That file is documented as an exact mirror copy, but its original feature
mapping and upstream transformation history are unresolved. The [Phase 9A
provenance report](phase9a_dataset_provenance.md) already investigates OpenML
1104 and concludes that it is a related, non-identical representation of the
same Golub benchmark. It is not an independent validation cohort.

This assessment therefore asks whether a separately collected cohort can
support a **replication of the leakage comparison**. Because the local
features are anonymous and have no verified mapping, none of the candidates
can support applying the exact selected features from the 72-sample matrix to
another cohort. A suitable study would rerun the prespecified safe and
deliberately leaky feature-selection comparison within the new cohort and
report it as an independent replication. Candidate datasets must not be
merged with the Stanford/Golub samples.

## Candidate comparison

| Candidate | Samples and ALL/AML counts | Assay and features | Preprocessing and metadata | Assessment |
|---|---|---|---|---|
| [MILE Stage II, GSE13164](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13164) | GEO lists 1,152 QC-passing samples. The original MILE paper reports 693 acute-leukemia cases in its independent stage-II validation set; summing its eight ALL subtype rows gives **436 ALL**, and its six AML rows gives **257 AML**. | Roche AmpliChip Leukemia custom expression array; GEO sample tables contain 1,480 probe-set rows. The paper describes 1,457 disease-related probe sets and 23 controls. The design is based on Affymetrix HG-U133 Plus 2.0, but it is a distinct custom panel, not the original platform. | The GEO sample record includes leukemia class, tissue source, treatment status, and processed signal fields. The study reports standardized preparation and summarization/quantile normalization; the archived `VALUE` is DQN3. The public sample record does not establish that every clinical or study-center field needed for grouped evaluation is available. | **Best candidate for independent replication**, because MILE explicitly calls Stage II a prospectively collected independent validation cohort. It is still a different, smaller feature panel; feature mapping and center metadata need review before protocol lock. |
| [MILE Stage I, GSE13159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159) | 2,096 QC-passing samples across 18 diagnostic classes. The original subtype table sums to **750 ALL** and **542 AML**. | Affymetrix HG-U133 Plus 2.0 (`GPL570`), 54,675 array probe-set rows in the GEO platform record. Probe-to-gene annotation exists, but probe multiplicity and annotation version require a fixed mapping rule. | The original study reports probe-set summarization and quantile normalization. GEO includes processed sample data and lists a 9.3 GB raw CEL archive. It combines 11 centers; the original study reports center-level class counts and notes center contributions vary by expertise. Whether center identifiers can be joined to every GEO sample must be verified. | Strong high-dimensional replication candidate and closer in scale to the original task. Less attractive than Stage II for the primary independent validation because the study is retrospective/multicenter and site-level grouping information is not yet confirmed. Could be a secondary analysis if center metadata are recovered. |
| [Childhood acute leukemia, GSE7186](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7186) | 121 consecutive childhood leukemia samples: **98 ALL** (87 B-lineage, 11 T-ALL) and **23 AML**. GEO also lists six normal samples; exclude them from an ALL-versus-AML task. | Swegene Human 27K RAP platform (`GPL4861`), a spotted DNA/cDNA array. Feature positions are described as coming from the file; mapping and effective unique-feature count need inspection. | GEO provides class, tissue, treatment, and assay-protocol metadata and a raw text archive. The series points to a 2005 protocol for laboratory processing; a complete reproducible normalization recipe for the archived matrix was not established in this desk review. | A real independent binary cohort, but much smaller, pediatric-only, imbalanced (98:23), and on a different platform. Useful as a sensitivity replication, not the preferred primary candidate. |

Counts for MILE Stage I and II are based on the gold-standard subtype rows in
the original MILE paper, not on a new download or reclassification. For Stage
II, 436 + 257 = 693; the other 459 GEO samples represent non-acute classes and
are not part of the proposed binary task. For Stage I, 750 + 542 = 1,292;
the remaining samples belong to other leukemia/MDS/control classes.

## Recommended candidate: MILE Stage II (GSE13164)

The [MILE original study](https://doi.org/10.1200/JCO.2009.23.4732) describes
two stages. Stage I used whole-genome Affymetrix arrays; Stage II was a
prospective, blinded validation on a separately collected independent patient
cohort using the Roche AmpliChip Leukemia custom array. Of 1,191 collected
Stage-II profiles, 1,152 passed the study's technical quality criteria. The
original paper reports 693 acute leukemia samples across eight ALL and six
AML subtypes in the validation set. GEO's [GSE13164
record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13164) provides
the public series, processed sample tables, and CEL archive listing. An
example [GEO sample record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM332229)
shows a leukemia-class label, tissue source, untreated status, and processed
expression values.

This is a good **cohort-level replication** candidate, not a compatible
extension of the existing 72-row matrix:

- It has over nine times as many acute cases as the current dataset, with both
  ALL and AML represented, and the original study used completed diagnostic
  workups to assign its subtype labels.
- Its 1,480 measured probe-set rows (including controls and tiled probes) are
  not the 7,128 anonymous positions in the Stanford file. The studies cannot
  share feature identities or be pooled by column position.
- It remains high-dimensional for the 693-case binary subset, but its custom
  panel is substantially smaller and more targeted than a whole-genome array.
  This changes the feature-selection problem and must be stated when comparing
  replication results.
- Because Stage II is prospectively collected and explicitly independent of
  MILE Stage I, it is a strong independent patient cohort. The local 72-sample
  file lacks patient identifiers, however, so direct person-level overlap
  checks against that file are impossible. The difference in source study and
  cohort supports independence, but an exact subject-level comparison cannot
  be made from the current repository.
- MILE involved multiple centers. Before choosing random folds, inspect all
  GEO characteristics and original supplementary metadata to determine
  whether center identifiers can be joined to samples. If possible, report a
  center-held-out sensitivity analysis; do not assume ordinary stratified
  folds measure transport to a new institution.

For scientific compatibility, preserve the existing project comparisons as
separate analyses: 50 selected features for the main safe-versus-leaky
performance comparison and 100 features across 20 resamples for feature
stability. Do not pool the candidate's scores or stability values with the
Stanford/Golub results. Before any run, predefine how the 23 control probes,
tiled/repeated probes, probe-to-gene mapping, missing values, and any
ambiguous diagnostic labels will be handled. Feature selection must remain
inside each training fold in the safe condition; all-label selection should
remain only the explicitly leaky diagnostic control.

## Access, licensing, and reproducibility

GEO makes these public series and processed sample tables accessible without
login. That is not an unqualified license. NCBI's [GEO
disclaimer](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) says NCBI
places no restrictions on GEO data use/distribution but warns that submitters
may claim patent, copyright, or other intellectual-property rights and that
NCBI cannot grant unrestricted permission on their behalf. No explicit
standard license for the GSE13164 data was found in the reviewed record. The
associated GPL7473 platform is described as a custom-commercial array, which
is a platform designation, not a data license. Obtain and document the
applicable rights for research analysis and any planned redistribution of raw,
processed, or derived data before sharing files or submitting a manuscript.

The same GEO rights caveat applies to GSE13159 and GSE7186. Their records
support public access, but no explicit data license was verified in this
assessment. Do not place raw or processed candidate data in this repository
until the rights and venue requirements are resolved. GEO's [sample
metadata](https://www.ncbi.nlm.nih.gov/geo/info/faq.html) and each sample's
data-processing fields should be archived as metadata references; record the
exact accession, retrieval date, file hashes, platform annotation version,
class inclusion rules, and preprocessing used if an analysis is later
approved.

## Other leads considered

- **OpenML leukemia dataset 1104:** already audited in Phase 9A. It shares the
  72-sample ALL/AML benchmark identity and Golub citation with the current
  study, but its values, sample ordering, and feature count differ. It is a
  different representation of the same benchmark, not an independent cohort.
- **GSE13204:** GEO identifies this as the MILE super-series containing the
  Stage-I and Stage-II subseries. It is not a third independent cohort; use
  the stage-specific accessions to avoid mixing them.
- **GSE13425 / GSE13351:** the GEO record describes 190 newly diagnosed ALL
  cases and an independent 107-case ALL validation cohort, but these are
  ALL-only and cannot provide the project's binary ALL-versus-AML comparison.

## GO / NO-GO recommendation

**CONDITIONAL GO** for an independently reported replication using GSE13164.
It is the clearest cohort-level validation candidate because the primary MILE
publication identifies Stage II as a prospective independent validation
cohort and supplies subtype counts for both ALL and AML. It offers a
sample-level class label and processed measurements, and its 1,457 disease
signal probes retain a high-dimensional setting for the 693-case subset.

**NO-GO for analysis, data redistribution, or publication claims at this
stage** until the project has:

1. Confirmed research-use and publication/redistribution rights with the
   dataset steward or target venue requirements.
2. Audited all sample labels, excluded non-acute cases, and reconciled the
   included count to 436 ALL and 257 AML from the published Stage-II table.
3. Determined availability of center/site identifiers and their suitability
   for grouped sensitivity evaluation.
4. Audited the GPL7473 feature table, tiled-probe structure, gene mapping,
   missingness, and the exact definition of the archived processed signal.
5. Prespecified a separate replication protocol and retained the existing
   Stanford/Golub results unchanged.

If Stage-II feature or rights checks fail, GSE13159 is the next candidate for
a high-dimensional replication, contingent on recovering usable center labels
and resolving its data-use terms. GSE7186 is a smaller, pediatric-only
sensitivity cohort. None should be substituted silently for the current data
or merged into it.

## Sources

1. [NCBI GEO, GSE13164: MILE study Stage 2 data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13164).
2. [NCBI GEO, sample GSM332229](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM332229), including sample characteristics and processed signal fields.
3. Haferlach T, et al. [Clinical utility of microarray-based gene expression profiling in the diagnosis and subclassification of leukemia: report from the International Microarray Innovations in Leukemia Study Group](https://doi.org/10.1200/JCO.2009.23.4732). *J Clin Oncol.* 2010;28(15):2529–2537.
4. [NCBI GEO, GPL7473: Roche AmpliChip Leukemia Custom Microarray](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL7473).
5. [NCBI GEO, GSE13159: MILE study Stage 1 data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159); [GPL570 platform](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570).
6. [Comprehensive Analysis of MILE Gene Expression Data Set Advances Discovery of Leukaemia Type and Subtype Biomarkers](https://pmc.ncbi.nlm.nih.gov/articles/PMC5366179/). *Interdiscip Sci Comput Life Sci.* 2017;9:24–35. Used here as a secondary source for the explicit collapsed ALL/AML counts and the reported preprocessing/batch analysis of Stage I.
7. [NCBI GEO, GSE7186: 121 childhood acute leukemias](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7186); [Andersson et al. (2007)](https://doi.org/10.1038/sj.leu.2404688).
8. [NCBI GEO, GEO disclaimer](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) and [GEO FAQ](https://www.ncbi.nlm.nih.gov/geo/info/faq.html).
9. [NCBI GEO, GSE13425](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13425) and [GSE13351](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13351), ALL-only discovery and validation cohorts.
