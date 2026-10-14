# 0014. Longitudinal PRAD BCR is a separate cohort with record-level evidence

## Status

Accepted.

## Context

Recorded 2026-09-25. Extends the available endpoints without changing the
`tcga-prad` CDR-PFI cohort or rewriting ADR 0008.

The historical Smith20 TCGA analysis converted
`biochemical_recurrence == "YES"` to an integer. Blank strings therefore
became zero, yielding 433 patients and 51 capped events. The original XML
marks all 54 blanks in that analysis population as `Not Available`, not NO.
Five of those patients have dated BCR in follow-up records. Longitudinal
inspection also recovers events in patients whose enrollment status was NO.

The completed independent R/Python audit found 500 source patients, 109 with
explicit BCR evidence, 102 timed events, seven untimed events, 383 supported
censored observations and eight other exclusions. There are 485 usable
clinical endpoints and 97 events after the 60-month cap. The historical RDS
expression snapshot intersects at 431 patients/92 capped events, but was
already restricted by the old clinical shortcut. It cannot define the future
production expression universe.

## Decision

Introduce `clinical_source: tcga_bcr` and a separate `tcga-prad-bcr` spec.
Read saved TCGA Clinical Supplement / BCR XML containing enrollment and every
follow-up, independently of expression. The clinical source is not CDR, and
neither PFI nor general new-tumor events are relabeled as BCR.

For each patient:

1. Explicit BCR is YES, a biochemical event/progression type, or a populated
   biochemical recurrence day field. Choose the earliest finite positive
   first-BCR day across records. Second/third dates establish evidence only;
   they do not substitute for the first date. Untimed BCR is excluded even
   if a negative observation exists. A populated but invalid date also cannot
   license negative fallback.
2. Without BCR evidence, an explicit NO BCR/new-tumor-event observation
   supports censoring at that SAME record's contact date. Choose the latest
   finite positive supported date. A later unknown cannot extend follow-up.
3. Other cases are excluded. Blank fields, tumor-free status, treatment
   response, information flags, absence of treatment, locoregional recurrence,
   distant metastasis, and new primary tumors alone do not establish a
   negative BCR observation.
4. Convert days using exactly /30.44. Cap at 60 months. Events strictly after
   60 months become censored; an event exactly at 60 remains an event.

The ledger retains all included and excluded patients, original record values
and procurement statuses, evidence, candidate dates, selected record/day,
reason, and per-file checksum. Positive and negative observations together
are flagged for review (they may represent legitimate longitudinal change,
not necessarily an error); differing first-BCR dates are separately flagged.
Malformed source schemas, duplicate patient XML versions, and duplicate record
identities fail rather than silently selecting one.

The XML snapshot checksum is SHA-256 over the newline-joined, sorted list of
per-file checksum strings (`sha256:<hex>`), without a trailing newline.
Relocating or flattening the directory does not change the pin. Per-file paths
and hashes remain in the report. The bundled spec pins the successfully
verified clinical snapshot and the actual 483-patient cached STAR build.
The golden reference records the expression manifest checksum, population hash
and 97 capped events; none of these is inferred from the historical RDS.

Expression continues through the existing GDC STAR workflow, primary-tumor
selection, patient deduplication, symbol harmonization and scale handling.
Clinical matching precedes all scoring/standardization. Production never loads
the historical expression RDS; that file is an optional checksum-gated
regression input only. No new HR, patient-count or event-count production
expectation is inferred from that RDS. The production sample-count expectation
is 483, pinned only after an actual full cached STAR build succeeded.

## Usage and reporting

Provide the original XML snapshot locally, then copy the bundled spec to a
working configuration and change `clinical_path` to the XML directory. The
source data are not redistributed. Do not change the clinical checksum merely
to accept a partial directory or another snapshot. Review the source and counts
before deliberately choosing a new pin.

```sh
hypoxiapipe cohort build /path/to/local-bcr.yaml --out out/prad-bcr --signature smith20 --offline
```

`--offline` requires the usual cached GDC manifest and STAR expression files;
remove it only when downloading those files is intended. Relative clinical
paths resolve against the invocation directory, as for the existing CDR spec.
A missing or mismatched clinical source fails before expression acquisition.

`cohort.json` retains the BCR ledger in clinical provenance and the build's
`tcga.endpoint` report. Source counts (`n_source`, `n_explicit_bcr`,
`n_timed_bcr`, `n_untimed_bcr_excluded`, `n_supported_censored`,
`n_unknown_excluded`, `n_before_expression_match`, `n_events_before_cap`,
`n_events_after_cap`, `n_censored_by_cap`) always describe the entire clinical
snapshot. `n_censored_by_cap` counts events converted to censored observations;
`n_followup_capped` counts already-censored observations truncated at 60.
`n_unknown_excluded` includes negative observations without a usable contact
date; `n_no_negative_evidence` and `n_negative_without_time` separate the causes.

`n_after_expression_match` is null in a clinical-only derivation and populated
by the TCGA join. `n_matched_events_before_cap`, `n_matched_events_after_cap`,
and `n_matched_censored_by_cap` describe the expression intersection. A patient
ledger entry also records `matched_expression` after joining. Existing cohort
QC reports the final cohort and signature coverage.

Optional real-data regressions require the explicit local inputs:

```sh
HYPOXIAPIPE_BCR_XML=/path/to/xml \
HYPOXIAPIPE_BCR_EXPRESSION_RDS=/path/to/tcga_prad_expression.rds \
pytest -q tests/test_tcga_bcr.py
```

The RDS test requires Rscript, verifies its exact SHA-256 before reading its
sample IDs and gene names, and checks 431/96/92 and Smith20 20/20. The
clinical-only regression checks 500/485/102/97 and the clinical checksum.
Ordinary offline CI uses synthetic equivalents and skips only these optional
real-data regressions when inputs are unavailable.

## Consequences

The historical 433/51 result and old CDR-PFI cohort remain reproducible and
unmodified. They are distinct endpoints/populations, not aliases for this
cohort. The full longitudinal analysis of the saved RDS yields 431/92 and
cannot be described as reproducing the old 433/51 analysis.

Source evidence makes cohort metadata larger. Retaining excluded patients is
intentional: a final clinical table alone cannot explain losses. Full-source
and matched counts must not be mixed in downstream reports. Using current
STAR TPM rather than the saved historical expression transform can also change
scores independently of clinical adjudication.

No model persistence, CoxNet serving, prediction endpoint, MLflow or deployment
is introduced.

## What would change this

A new clinical snapshot or adjudication policy requires an explicit source
review, new audit and versioned endpoint decision. A new expression release
requires reporting the actual intersection, not imposing 431 or 433 patients.
