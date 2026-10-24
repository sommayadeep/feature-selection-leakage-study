# Phase 9A: Dataset Provenance and Licensing Investigation

## Executive conclusion

The project’s raw dataset is **verified as an exact byte-for-byte copy of the
Stanford-hosted `leukemia_big.csv` mirror** maintained under Trevor Hastie’s
CASI files. That page identifies the file as gene-expression measurements from
72 leukemia patients (47 ALL and 25 AML), states that the data arise from the
Golub et al. (1999) Science paper, and describes the file as a 7,128 x 72
matrix.

The local file’s immediate public source can therefore be cited as:

> Hastie, T. CASI files, “Leukemia data,” `leukemia_big.csv`,
> <https://hastie.su.domains/CASI_files/DATA/leukemia.html>.

The underlying study citation is:

> Golub, T. R., Slonim, D. K., Tamayo, P., Huard, C., Gaasenbeek, M.,
> Mesirov, J. P., Coller, H., Loh, M. L., Downing, J. R., Caligiuri, M. A.,
> Bloomfield, C. D., & Lander, E. S. (1999). Molecular classification of
> cancer: class discovery and class prediction by gene expression monitoring.
> *Science*, 286(5439), 531–537.
> DOI: <https://doi.org/10.1126/science.286.5439.531>.

The file-level match does **not** resolve the original sample identities,
feature mapping, upstream transformation, or licensing terms. The Stanford
page explicitly states that the genes in the large dataset were transformed
and that the exact transformation has been lost. It also states that the
correspondence to the smaller dataset is unknown. No license or redistribution
permission is stated on the Stanford page.

**Recommendation:** do not make biological or generalization claims, and do
not treat publication/redistribution permission as established, until the
data-use terms and original feature/sample provenance are clarified. A
narrowly scoped methodological paper may be possible with an explicit
provenance limitation if the venue accepts the Stanford mirror as the cited
data source, but the project should resolve the licensing question first.

## 1. Local evidence inspected

### Files and documentation

Inspected:

- [`src/prepare_leukemia.py`](../src/prepare_leukemia.py)
- [`reports/dataset_audit.md`](./dataset_audit.md)
- [`reports/project_inventory.md`](./project_inventory.md)
- [`reports/methodology_audit.md`](./methodology_audit.md)
- [`reports/phase8_scientific_audit.md`](./phase8_scientific_audit.md)
- All project documentation and metadata files found outside `.venv`.
- Raw and processed dataset headers and file metadata.

No project `README`, data citation file, DOI record, accession record,
download manifest, license file, `CITATION.cff`, or source URL record was
present. The existing reports correctly recorded the provenance gap before
this investigation.

### Local file structure

`data/raw/leukemia_big.csv`:

- 7,129 rows x 72 columns.
- No header row.
- First row contains the 72 class labels.
- Remaining 7,128 rows contain numeric measurements.

`data/processed/leukemia.csv`:

- 72 rows x 7,129 columns.
- 7,128 generated feature columns plus `target`.
- The local preparation script transposes the raw matrix, assigns
  `gene_0001` through `gene_7128`, and appends the labels.

The preparation script does **not** normalize or otherwise transform numeric
values. Any upstream numeric transformation is already present in the raw
file.

### Local hashes

| File | SHA-256 |
|---|---|
| `data/raw/leukemia_big.csv` | `35e84928da625da0787efb31a451dedbdf390e821a94ef74b7b7ab6cab9466d4` |
| `data/processed/leukemia.csv` | `21ea525a9a66b19945d2f7d2788c9d83373eaafd4ffdafe00a3f959a6b677fce` |

These hashes agree with the values previously recorded in
[`dataset_audit.md`](./dataset_audit.md). The processed hash identifies the
current local transpose/name-generation output; it is not a source-dataset
hash.

## 2. Candidate source investigation

### Candidate A: Stanford CASI `leukemia_big.csv` — verified exact file source

**Status: VERIFIED for file identity and immediate public mirror.**

Source page:
<https://hastie.su.domains/CASI_files/DATA/leukemia.html>

Direct file:
<https://hastie.su.domains/CASI_files/DATA/leukemia_big.csv>

Evidence:

1. The direct download has SHA-256
   `35e84928da625da0787efb31a451dedbdf390e821a94ef74b7b7ab6cab9466d4`,
   exactly matching the local raw file.
2. A byte comparison returned equality, not merely equal dimensions or
   summary statistics.
3. The Stanford description reports 72 leukemia patients, 47 ALL, 25 AML,
   and a 7,128 x 72 matrix, matching the local raw orientation.
4. The Stanford page states that the data arise from the Golub et al. (1999)
   Science paper.
5. The public directory lists `leukemia_big.csv` as the 6.1 MB large
   leukemia file, consistent with the local file size of 6,364,027 bytes.

Contradictory or limiting evidence from the same authoritative mirror:

- The page says the genes in the large dataset were transformed and that the
  exact transformation was lost.
- The page says the columns of the large and small datasets are in different
  orders.
- The page says the correspondence between the 3,571-feature small dataset
  and the 7,128-feature large dataset is unknown.
- The page does not provide patient identifiers, sample metadata, probe
  mapping, preprocessing parameters, or a license statement.

Thus the local raw file’s **immediate mirror source is verified**, but the
complete original provenance of the transformed matrix is not.

### Candidate B: Golub et al. (1999) publication — verified cited study, not a
byte-level source record

**Status: VERIFIED as the study cited by the Stanford mirror and OpenML;
UNRESOLVED as the exact upstream file/provenance chain.**

Bibliographic record:

- Europe PMC record for PMID 10521349:
  <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:10521349&format=json>
- DOI: <https://doi.org/10.1126/science.286.5439.531>

The bibliographic record reports the title, author list, *Science* volume 286,
issue 5439, pages 531–537, and publication year 1999. The Stanford page
directly says that its leukemia data arise from this landmark paper.

The local file cannot be claimed to be the original supplementary file from
the paper solely from dimensions or class counts. The Stanford page documents
an intervening transformed large matrix and explicitly says that the
transformation history is lost.

### Candidate C: OpenML dataset 1104 — plausible related benchmark, not an
exact local-file match

**Status: VERIFIED as an accessible related OpenML record; NOT VERIFIED as the
exact project dataset.**

OpenML metadata:
<https://www.openml.org/api/v1/json/data/1104>

OpenML download:
<https://openml.org/data/v1/download/53987/leukemia.arff>

The OpenML record identifies dataset `leukemia`, version 1; cites the Golub
paper; identifies an ARFF file; reports file ID 53987; and gives a metadata
licence value of `Public`. The downloaded ARFF contains 72 observations, 7,129
numeric probe attributes, and a `CLASS` attribute. It contains the same two
class labels and the same total class counts (47 ALL, 25 AML).

Direct comparison with the local raw file found important differences:

| Property | Local Stanford-mirror file | OpenML 1104 download |
|---|---:|---:|
| Numeric feature count | 7,128 | 7,129 |
| Matrix orientation | 7,128 x 72 | 72 x 7,129 plus class |
| Feature values | Approximately -3.527 to 3.808 | -28,400 to 71,369 |
| Label/sample ordering | Different ordering | Different ordering |
| Exact values after orientation | Not equal | Not equal |

The downloaded OpenML ARFF had SHA-256
`b10d7bfd33b8b881cbf3a1a30ea3d8ebeb0fa39d741f35b6d82d59daa5fa5ed7`.
OpenML metadata reports MD5
`338f0779a7c2e0ad62f98d987b43edae`.

The matching paper citation and class/sample counts make OpenML 1104 a
plausible related representation of the Golub benchmark. The one-feature
difference, distinct values, and distinct ordering prevent a claim that it is
the exact local source file. It may represent a different release,
preprocessing stage, or feature set.

### Candidate D: public GitHub mirror — corroborative copy, not authoritative

**Status: VERIFIED exact byte copy; NOT an authoritative provenance or license
source.**

The repository
<https://github.com/Claudmj/Microarray-analysis-with-Bayesian-networks>
contains a file at
<https://github.com/Claudmj/Microarray-analysis-with-Bayesian-networks/blob/4bf6640fa6ffcffb60922269eded83ace0678072/Leukemia/leukemia_big.csv>.

The raw GitHub file has the same SHA-256 as the local file:
`35e84928da625da0787efb31a451dedbdf390e821a94ef74b7b7ab6cab9466d4`.
Its repository README describes the file as already-preprocessed and its
feature-selection code refers to the dataset as the Golub dataset.

This is useful corroboration that the bytes circulate publicly, but it does
not establish the original source, ownership, license, or permission to
redistribute the data. A GitHub copy cannot substitute for an authoritative
data-provider license statement.

## 3. Original publication and accession

### Publication

The publication citation is sufficiently verified for citation purposes:

Golub et al. (1999), “Molecular classification of cancer: class discovery and
class prediction by gene expression monitoring,” *Science*, 286(5439),
531–537, DOI
<https://doi.org/10.1126/science.286.5439.531>.

The citation is independently supported by the Europe PMC bibliographic
record and the Stanford data page.

### Accession

**UNRESOLVED for the exact local transformed matrix.**

OpenML dataset ID 1104 and file ID 53987 are verified identifiers for the
OpenML ARFF candidate. They should not be reported as the accession of the
local Stanford CSV without qualifying that the files differ.

No GEO accession, ArrayExpress accession, Broad Institute file identifier, or
original supplementary-file identifier was found in the local project or on
the Stanford data page. The OpenML metadata itself says “Source: Unknown -
Date unknown,” despite citing the Golub publication.

## 4. Feature identifier mapping

### Local feature names

**VERIFIED.** `gene_0001` through `gene_7128` are generated positional names
created by [`prepare_leukemia.py`](../src/prepare_leukemia.py). They are not
source-provided gene symbols, probe IDs, accession numbers, or verified
biological identifiers.

### Original feature identifiers

**UNRESOLVED.** The Stanford large CSV has class labels as column names but
does not provide a feature-name row or mapping for the 7,128 numeric rows.
The Stanford page explicitly warns that the large matrix’s genes have been
transformed and that the exact transformation is lost.

OpenML 1104 provides probe-like feature names such as
`AFFX-BioB-5_at` and `U58516_at`, but the downloaded OpenML file has a
different feature count and different values. Those names cannot be assigned
to local `gene_####` columns by position or by guesswork.

**Conclusion:** no verified mapping exists for the local features. Selected
features must remain positional/anonymized predictors; they must not be
described as genes, probes, pathways, or biologically meaningful loci.

## 5. Sample metadata and preprocessing history

### Sample metadata

**VERIFIED locally:** the raw file contains class labels only. The processed
file contains those labels and generated feature names. There are no patient
IDs, clinical covariates, collection centers, dates, treatment information,
batch variables, family/relatedness indicators, or duplicate-sample keys.

**UNRESOLVED:** whether all 72 columns represent independent patients in the
clinical sense, whether any samples are related or duplicated, and whether
technical batches correlate with class.

### Numeric preprocessing history

**VERIFIED:** the local preparation script only transposes the raw matrix,
generates positional names, and writes the processed file. It does not create
the observed numeric values.

**UNRESOLVED:** the upstream normalization, scaling, transformation,
background correction, probe filtering, and any sample-level preprocessing
used before the Stanford mirror was published. The Stanford page explicitly
states that the large-matrix gene transformation is lost.

The current experiment’s fold-local imputation, scaling, and discretization
are reproducible transformations applied after the local file is loaded.
They do not recover or document the upstream history of the dataset.

## 6. Licensing and permitted use

### Stanford mirror

**Status: UNRESOLVED.**

The Stanford data page provides download instructions and a scientific
description but no explicit license, copyright statement, terms of use,
research-use permission, or redistribution permission. Public
downloadability is not evidence of a license.

### OpenML representation

**Status: PARTIALLY DOCUMENTED, not sufficient for the local file.**

OpenML dataset 1104 reports `licence: "Public"` in its API metadata:
<https://www.openml.org/api/v1/json/data/1104>.

This is a metadata value, not a standard license such as CC BY, and it does
not establish that the Stanford mirror or the transformed local file may be
redistributed. It also does not resolve the rights associated with the
underlying clinical/microarray data.

### Publication rights

**UNRESOLVED.** The Golub paper citation establishes scholarly provenance, not
the right to redistribute the raw matrix or publish derived data under a
particular license.

The project should distinguish:

1. Permission to download and use a public file for research.
2. Permission to publish analyses derived from it.
3. Permission to redistribute the original matrix.
4. Permission to redistribute a transformed or feature-selected derivative.

None of these permissions is documented locally for the exact Stanford
mirror file.

## 7. Remaining uncertainties

1. The precise upstream source file that produced the Stanford transformed
   matrix is not identified.
2. The transformation applied to the 7,128 features is undocumented and,
   according to the Stanford page, lost.
3. The original probe/gene identifiers and their order are unavailable.
4. The 72 sample identities and independent-patient status are unavailable.
5. Batch, laboratory, collection-site, and related-sample metadata are
   unavailable.
6. No exact-data license or redistribution terms were found.
7. OpenML 1104 is related by dimensions/classes/citation but is not an exact
   value-level match and must not be silently substituted.
8. The original paper’s supplementary data cannot be treated as identical to
   the Stanford transformed matrix without an explicit source chain.

## 8. Publication recommendation

**Do not proceed as if provenance and licensing were resolved.**

Before manuscript submission, obtain one of the following:

1. An authoritative source record for the Stanford mirror with explicit terms
   permitting the intended research and publication use; or
2. A documented decision from the target venue/editor that citation of the
   Stanford public mirror plus transparent disclosure of the missing
   transformation and license is acceptable.

If publication proceeds without resolving the upstream mapping, the paper
must be restricted to a methodological demonstration on an anonymized
positional matrix. It must cite the Stanford mirror and Golub et al., state
that the feature identifiers are synthetic, state that the upstream
transformation is undocumented, and omit biological interpretation.

The project must not:

- Replace the current dataset with OpenML 1104 without a new approved
  protocol and experiment set.
- Assign OpenML probe names to local `gene_####` columns.
- Claim a GEO/Broad accession that was not verified.
- Infer a standard license from public accessibility or the OpenML
  `Public` metadata value.
- Redistribute the original or derived matrix until permission is clarified.
