# 0007. Preprocessing is fitted inside every fold and frozen

## Status

Accepted.

## Context

Standardising features before splitting into folds lets held-out samples
influence the statistics used to train. It never errors. The resulting optimism
is small — often a percentage point or two of C-index — which is exactly what
makes it dangerous: too small to look wrong, large enough to matter when
comparing models.

The same problem recurs at deployment. A model trained on standardised features
must be applied with the *training* statistics, not the new cohort's own, or the
prediction depends on who else happened to be measured alongside the patient.

## Decision

`ReferenceScaler` separates `fit` from `transform`. Every CV fold — inner and
outer — fits its own scaler on its training half only, and a fitted model
carries its scaler so external cohorts are scored against frozen constants.
Refitting a fitted scaler raises, because in a nested loop that almost always
means a fold boundary was crossed.

`assert_no_leakage` tests the property directly: perturb only the held-out
samples, and the training scaler's checksum must not move. A second test
confirms the guard itself fires when a scaler is fitted on everything, so it
cannot quietly stop checking.

## Consequences

More scalers are fitted than strictly necessary, which is negligible at this
scale.

Frozen statistics are what make a single-sample prediction well defined at all,
which determines the shape of the planned scoring service: a `/score/reference`
endpoint applying stored constants, and a 422 for single-sample requests against
methods that cannot validly produce one.

## What would change this

Nothing. Feature selection, if it becomes part of the model, must move inside
the fold boundary on the same reasoning.
