# 0005. Cohorts are restricted to their analysis set before any standardisation

## Status

Accepted.

## Context

Per-gene z-scoring is cohort-relative: a gene's score depends on the mean and
standard deviation across the samples in the matrix. Standardising the full
matrix and then dropping patients with no outcome data is therefore a different
analysis from dropping them first, because the discarded samples still shaped
the scores of the ones kept.

This is not hypothetical. A canary value in the Cambridge cohort moved between
1.767 and 1.798 depending purely on whether standardisation ran before or after
subsetting. Nothing errored; the number simply differed.

## Decision

`build_cohort` narrows every cohort to its analysis population — the samples
with a usable endpoint — before returning it, and records a `population_hash`
over the sample set. Standardisation and scoring happen downstream of that,
always. The narrowing is a recorded provenance step with a count of what was
dropped.

## Consequences

A cohort object is bound to one endpoint. Analysing the same cohort under a
different endpoint means building it again, which is more work but produces two
distinct, separately hashed populations rather than one object quietly meaning
two things.

The `population_hash` travels into the manifest, so any score can be traced to
the exact sample set it is relative to.

## What would change this

Nothing about the ordering. If multiple endpoints per cohort become common,
`Cohort` might hold several analysis sets explicitly — but each score would
still name the population it belongs to.
