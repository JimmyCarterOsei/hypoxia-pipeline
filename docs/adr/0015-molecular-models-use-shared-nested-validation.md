# 0015. Molecular models share repeated nested validation on the locked STAR BCR cohort

## Status

Accepted.

## Context

This decision governs Phase 9C.2 development; no production model is designated.

Phase 9C.1b established that the audited STAR/log-TPM population preserves a
Smith20 association but is not interchangeable with historical VST expression.
Time-dependence diagnostics justify an explicit non-PH comparator. Apparent
full-cohort fits cannot choose between a fixed biological signature, learned
gene coefficients and a nonlinear model.

## Decision

Extend the existing `modeling.train.nested_cv` engine, retaining its frozen
`ReferenceScaler`, rather than create a second evaluator. Use five repeated
five-fold stratified outer splits and five-fold stratified inner splits. Sort
patient IDs, align labeled outcomes, and precommit every split and the modest
parameter grids before any fit. Use identical outer and inner partitions for
all models. Inner candidates must complete every fold; an incomplete candidate
is never averaged over its successful subset. Exact ties retain the first
prespecified candidate. All-zero penalized fits are valid constant-risk fits;
convergence warnings are failures.

M0 is the fixed Smith20 mean of 20 training-standardized genes, with the
composite standardized by its training sample SD. Its single Cox coefficient
is fitted in R survival with Efron ties through the existing JSON bridge; a
new frozen-score action does not standardize the covariate again. There is no
Python fallback (ADR 0006). M1 learns a new CoxNet predictive model from the
20 individual genes; its coefficients are never named Smith20. Its native
Breslow tie approximation is recorded. M2 is a survival forest using the same
20 genes. There are no clinical covariates or external tuning cohorts.

Every transform and tuning operation sees only its training partition. Missing
required genes, nonfinite input and ambiguous sample IDs fail. Additional genes
are ignored under the explicit feature contract. Export held-out predictions,
training statistics, all inner scores/failures, selected parameters and
coefficients. RSF importance is held-out permutation loss of concordance,
computed after tuning, never used for feature selection, and is not causal.

The primary metric is the equal-weight mean outer-fold Harrell concordance.
Do not rank risks from different fitted folds against each other. Also report
within-fold comparable-pair-weighted concordance, fold/repeat dispersion, and
paired patient-cluster bootstrap intervals using common multiplicity weights
across models and repetitions. These intervals condition on saved predictions;
they do not include re-fitting/training-selection uncertainty. Repeated folds
are not independent studies. Additional censoring-sensitive metrics are omitted
until an appropriate IPCW and time-support workflow is implemented and tested.

Only after all outer evaluations finish, tune and fit full-population models
for numerical diagnostics. Do not serialize fitted models, create deployment
bundles, or designate a winner. Independent validation requires a separately
reviewed, frozen training-domain transform; cohort-relative restandardization
of a future validation cohort would violate this contract.

## Consequences

One command, `hypoxiapipe model develop-molecular COHORT EMPTY_OUTPUT`, runs the
pinned Phase 9B STAR/BCR cohort through the prespecified design. This command
rejects other cohort hashes and endpoint-ledger discrepancies. R remains a hard
dependency. Runtime is higher than a single CV split, and the resulting
confidence intervals remain conditional rather than a full bootstrap of the
training procedure. Neither hyperparameters nor splits are revised after
viewing outer performance.

## What would change this

A new development cohort, a reviewed requirement for censoring-aware metrics,
or independent validation evidence can justify a separately versioned protocol.
Such changes require a new decision and a new precommitted run; outer results
from this run must not be used to retrospectively change its feature set,
parameter grid, splits or decision thresholds.
