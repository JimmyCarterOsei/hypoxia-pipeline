# hypoxia-pipeline

Reproducible transcriptomic signature scoring with content-addressed signature provenance.

## Why

A benchmark in the upstream research project was invalidated by a comparator gene list
that shared only **4 of 32 genes** with the published signature it was labelled as. The
error survived for months across three scripts because nothing verified gene-list
provenance — a comment citing the source paper sat directly above the wrong genes.

This package makes that class of error structurally impossible: signatures are
content-addressed YAML specs, and loading one verifies its SHA-256. An edited gene list
fails loudly instead of silently producing plausible numbers.

## Install

```bash
pip install -e ".[dev]"
```

## Use

```bash
hypoxiapipe sig list                      # bundled signatures + verification status
hypoxiapipe sig verify path/to/spec.yaml  # non-zero exit on checksum failure
hypoxiapipe sig hash spec.yaml --write    # re-hash after verifying against source
hypoxiapipe score matrix.tsv -s ragnum32  # score samples (genes x samples input)

hypoxiapipe cohort list                   # bundled cohort specs + pinned expectations
hypoxiapipe cohort inspect cambridge      # list the clinical columns GEO actually returns
hypoxiapipe cohort build cambridge -o out/cambridge -s smith20
hypoxiapipe cohort build tcga-prad -o out/prad -s smith20   # GDC manifest + STAR counts
hypoxiapipe cohort qc out/cambridge -s smith20   # re-QC a stored cohort, checksum verified

hypoxiapipe manifest show out/cambridge   # what went in, what came out, on which code
hypoxiapipe manifest verify out/cambridge # re-hash the outputs; non-zero exit on drift

hypoxiapipe model validate out/cambridge -s smith20      # per-SD Cox HRs, fitted in R
hypoxiapipe model validate out/cambridge -s smith20 --action quartile
hypoxiapipe model cv out/cambridge --estimator coxnet -o out/cv   # nested CV
```

`cohort build` runs the whole ingest path: fetch (cached), map probes to symbols via the
platform annotation, harmonise symbols onto the pinned HGNC release, infer or apply the
log2 transform, derive the survival endpoint, restrict to the analysis set, QC, and save.
It exits non-zero if QC fails or the cohort contradicts the expectations pinned in its spec.

## Cohort specs

A cohort spec pins an accession, its platform, how the endpoint is encoded, and what the
built cohort should contain:

```yaml
name: Cambridge
source: geo
accession: GSE70768
platform: GPL10558
endpoint:
  time_column: bcr_free_time
  event_column: bcr_status
  time_unit: months        # required - days and months look identical until the HRs move
  cap_months: 60           # administrative censoring, recorded as a provenance step
expect:
  n_samples: 111           # asserted at build time, not documentation
```

Two ordering rules are enforced rather than left to the caller. Symbol harmonisation runs
before scoring, because a signature listing `CYR61` scores nothing against a matrix
labelled `CCN1` and the coverage report calls the gene missing. Restriction to the
analysis set runs before any standardisation, because per-gene z-scores are relative to
the samples in the matrix — standardising the full matrix and then subsetting gives
different scores from subsetting first. Every built cohort carries the
`population_hash` its scores are relative to.

## Scoring methods

| method | definition | use |
|---|---|---|
| `rowmean` | per-gene z across samples, then mean | default; equal gene weighting |
| `median_z` | median raw expression, then z | legacy; dominated by high-expression genes |
| `weighted` | Σ(coefficient × per-gene z) | signatures published with Cox coefficients |

Yang28 **must** be scored `weighted`: it is directionally mixed, so unweighted averaging
cancels (verified: unweighted null; weighted HR 1.63, P=4e-4 in TCGA-PRAD).

## TCGA

A TCGA cohort is assembled from several hundred per-aliquot STAR-counts files, each a
UUID that means nothing on its own. `cohort build tcga-prad` queries the GDC files
endpoint, caches the manifest, fetches and hashes every file individually, then makes
three joins that each silently produce a wrong cohort if done carelessly:

- **aliquot → patient** — expression is per aliquot (`TCGA-CH-5761-01A-11R-1580-07`),
  clinical is per patient (`TCGA-CH-5761`); joining without truncating gives an empty
  intersection, truncating without deduplicating gives duplicate patients;
- **tumour vs normal** — adjacent normals sit in the same project and widen every gene's
  apparent dynamic range;
- **Ensembl versions** — `ENSG00000141510.16` changes suffix between GDC releases.

Losses at each stage are counted in the build report rather than inferred from a total,
and `tolerate_file_failures` is 0 by default: a cohort quietly missing samples is a
different cohort from the one the manifest describes.

### The PRAD endpoint problem

GDC clinical carries no usable recurrence outcome for prostate, and overall survival in
TCGA-PRAD has roughly a dozen events across ~500 patients — a model fitted on that returns
a hazard ratio, a confidence interval and a p-value that all look like results. The
pipeline therefore takes the endpoint from the **TCGA Clinical Data Resource** (Liu et al.,
*Cell* 2018) and uses **PFI**, the endpoint those authors recommend for PRAD.

Asking for OS or DSS in PRAD raises. Overriding requires `allow_discouraged=True` in code,
not a different string in a config file. The CDR table is a supplementary file from that
paper and is **not redistributed here** — point `clinical_path` at your own copy.

## Containers

Two images, not one:

| image | contents | entry point |
|---|---|---|
| `hypoxiapipe` | ingest, QC, harmonise, score, model, provenance | `hypoxiapipe` CLI |
| `hypoxiapipe-r` | R + `survival` + `jsonlite` and the worker | the worker itself |

```bash
make images
export HYPOXIAPIPE_R_COMMAND="docker run --rm -i ghcr.io/JimmyCarterOsei/hypoxiapipe-r:TAG"
hypoxiapipe model validate out/prad -s smith20      # survival runs in the R image
```

`HYPOXIAPIPE_R_COMMAND` replaces the command used to reach the worker. Because the
contract is one JSON request on stdin and one response on stdout, any transport that
honours it works — a local `Rscript`, a container, a remote shell — with no service, no
socket and no shared volume. That indirection is what the narrow contract bought.

A single fat container would be larger, slower to rebuild, and would make the
per-process containers in Phase 7 meaningless, since every process would run the same
image. The Python image installs no R; the R image installs no Python package, and both
run as non-root.

## Workflow

```bash
python tools/make_test_fixtures.py
nextflow run workflow/main.nf -profile test              # offline, ~20s

nextflow run workflow/main.nf -profile docker \
    --cohorts cambridge,stockholm,tcga-prad \
    --signatures smith20,ragnum32 \
    --actions cox_persd,quartile \
    --estimators coxnet
```

Cohorts fan out as a channel: each is built, validated and cross-validated independently
and in parallel, then collected. That is the only reason to use a workflow manager here
rather than a driver script — four cohorts on a laptop and four cohorts on AWS Batch are
the same DSL with a different `-profile`.

Containers are declared **per process**, not globally, which is what the split images from
Phase 6 were for. The `docker` and `singularity` profiles set `HYPOXIAPIPE_R_COMMAND` so
the survival call reaches the R image identically in every environment.

Failures terminate rather than retry: `-resume` makes re-running a long ingest cheap, so
silently retrying a failure is worse than stopping and showing it. Every run writes a
timeline, report, trace and DAG into `results/pipeline_info/`, alongside the per-step
manifests the package itself writes.

## Golden references

`tests/golden/tcga-prad.json` freezes the results of a verified real build — counts at
every assembly stage, the `population_hash`, signature coverage, and the fitted survival
estimates. `tests/test_golden.py` compares a freshly built cohort against it:

```bash
hypoxiapipe cohort build prad.yaml --out out/prad -s smith20
HYPOXIAPIPE_GOLDEN_COHORT=out/prad pytest tests/test_golden.py -v
```

They skip when no built cohort is available, which is the honest default — a regression
test that silently passes when it cannot run is worse than one that says it did not run.

No cohort data is stored: only counts, hashes and statistics, so the reference is
checkable without redistributing bulky or controlled files. `expr_checksum` is
deliberately *not* asserted — it moves whenever GDC re-releases the underlying files,
which is a data change rather than a regression, and a test that fails on a legitimate
upstream release trains people to ignore it. `population_hash` is asserted, because it
covers the exact sample set every score is relative to.

## Run manifests

Every `cohort build` writes a `manifest.json` recording the run: the accession and the
checksum of the bytes fetched, each signature and its registry hash, the symbol authority
and the hash of the mapping table, the package version, git SHA (with `-dirty`), container
digest where set, the parameters, and the hash of every file written.

`manifest verify` re-hashes those outputs and exits non-zero if any is missing or has
changed, so a stale result on disk is detectable rather than assumed current. The
manifest's own `manifest_hash` covers inputs and parameters but not outputs, which
accumulate as a run proceeds — it identifies *this run, so configured*.

The point is narrow and worth stating plainly: the failure this project came out of was
not a bug in anyone's code, it was an input that quietly stopped being what its label
said. A result with a manifest can be re-checked against its inputs a year later.

## Modelling

Survival estimation runs in **R** (`survival`) behind a JSON stdin/stdout contract; the
Python side handles data, folds and provenance. Reimplementing `survival` in Python would
cost weeks and be worse than the original, and a silent Python fallback would mean two
different estimators producing results with the same label — so R's absence is a clear
error, never a substitution.

Model selection uses nested cross-validation, and **preprocessing is fitted inside every
fold**. Standardising before splitting lets held-out samples shape the training features;
it never errors, and the optimism is small enough to look like a real result.
`ReferenceScaler` separates `fit` from `transform` so the constants that trained a model
are the ones any later cohort is scored against — which is also what makes a single-sample
prediction well defined, and why the planned service exposes `/score/reference`.

`assert_no_leakage` tests that property directly: perturb only the held-out samples and
the training scaler's checksum must not move.

## Decisions

The choices that shaped this pipeline are recorded in [docs/adr](docs/adr/), written as
they were made rather than reconstructed afterwards. Each states what was actually in
tension, what it costs, and what evidence would reverse it. The load-bearing ones:

- [0006](docs/adr/0006-survival-runs-in-r.md) — survival runs in R with **no** Python
  fallback, because two estimators under one label is the failure this project came from
- [0008](docs/adr/0008-prad-endpoint-is-cdr-pfi.md) — TCGA-PRAD refuses overall survival:
  ~12 events in 500 patients produces numbers shaped like results
- [0007](docs/adr/0007-preprocessing-frozen-inside-folds.md) — preprocessing frozen inside
  every fold, with the leakage property tested rather than asserted
- [0011](docs/adr/0011-duckdb-deferred.md) — DuckDB deliberately *not* built yet

## Status

Phases 1–4, 6 and 7 of 10 complete: package core (registry, scoring, CLI); ingest/QC/harmonise
(GEO and TCGA/GDC ingest, probe mapping, pinned HGNC symbols, endpoint derivation,
cohort store);
provenance (content hashing, run manifests, verification); modelling (frozen preprocessing,
coxnet, random survival forest, nested CV, R survival bridge); containers and CI
(split Python/R images, GHCR publishing, pluggable R transport); Nextflow DSL2
orchestration with an offline test profile that runs the full DAG in ~20s.
265 tests, all offline, plus golden-file regression tests against a verified
TCGA-PRAD build (GDC release 46.0). Decisions recorded in [docs/adr](docs/adr/).
MLflow/API and cloud execution follow.

DuckDB is deliberately not built yet: a metadata store that nothing queries is decoration,
so it lands with the reporting layer that reads from it.
