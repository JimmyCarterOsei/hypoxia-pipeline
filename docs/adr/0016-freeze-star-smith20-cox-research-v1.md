# 0016. Freeze STAR Smith20 Cox as the first research-use predictor

## Status

Accepted.

## Context

Phase 9C.2 gave fixed Smith20 Cox competitive repeated nested-CV discrimination.
CoxNet did not improve it; RSF was numerically higher, but paired uncertainty
did not establish superiority. Phase 9D.1 reproduced the fixed Smith20
association in three independent microarray cohorts. Those analyses used
cohort-relative scoring and do not validate a STAR-trained prediction rule.
Phase 9C.3a found incremental internal discrimination with clinical variables,
but ordinary clinical Cox was unstable in folds with no low-grade events.

## Decision

Designate the fixed Smith20 Cox model as the first **research-use production
candidate**, STAR-Smith20-Cox v1. Refit its 20 gene means and sample SDs, mean
composite and composite sample SD on all 483 pinned STAR/BCR development
patients. Fit one Cox coefficient in R survival with Efron ties and preserve
all numerical values as checksum-verified JSON. Inference uses only these
frozen training constants. It requires a declared GDC-compatible STAR TPM,
log2(TPM + 1), harmonised-symbol expression domain and all 20 genes.

M2 remains an experimental nonlinear/non-PH comparator. C1 remains a research
combined model. Neither is substituted for the fixed biological signature.
Legacy external microarray results support association under their own
cohort-relative scoring, but lack a validated transform into frozen STAR
feature space. They are not recorded as external validation of this predictor.

The bundle returns standardized Smith20, Cox linear predictor and relative
hazard, optionally with a deterministic training percentile. A baseline hazard
and calibration model have not been independently validated, so v1 does not
return an absolute recurrence probability.

## Consequences

The JSON bundle is inspectable and rejects checksum corruption, missing or
duplicated genes, nonfinite values and undeclared or incompatible expression
domains. Future samples never refit scaling. A relative hazard is a comparison
to the training reference, not a patient's chance of recurrence. No FastAPI
prediction endpoint or opaque executable serialization is introduced.

## What would change this

Independent prostate-tumour RNA-seq with compatible STAR TPM processing and a
usable recurrence endpoint could validate the frozen predictor. A prespecified
and independently validated cross-assay transformation could permit another
domain. Calibrated absolute probabilities would require separate validation
of the full risk and censoring model. A larger clinical analysis could revisit
C1 once sparse reference strata and variable provenance are resolved.
