# 0013. Single-sample scoring is refused, not approximated

## Status

Accepted.

## Context

A scoring service invites one obvious request: send a patient, get a score. It
is what a clinician would expect, what a demo wants, and what most such
services provide.

Every scoring method in this package is cohort-relative. `rowmean` and
`weighted` z-score each gene across the samples supplied; `median_z`
standardises the per-sample median across them. So a score is not a property of
a patient — it is a property of a patient *and the batch they arrived in*. The
same tumour scores differently depending on who else was on the plate.

With one sample there is no distribution at all, and the arithmetic is
undefined. Worse than undefined: it is easy to make it return a number. A
single-sample z-score of zero, or a NaN quietly filled, produces a well-formed
response that a downstream system would store, plot and act on. That is a more
dangerous failure than an error, because nothing about it looks wrong.

This is the same cohort-relativity that ADR 0005 handles at ingest and ADR 0007
handles at model fitting, met a third time at the service boundary.

## Decision

Two endpoints with different guarantees.

`/score/batch` scores a cohort relative to itself, and **refuses** submissions
below a cohort-sized threshold with 422. The error explains why, names the
threshold, and points at the endpoint that can answer the question.

`/score/reference` scores against per-gene statistics frozen from a named
cohort. This is valid for a single sample, because the distribution comes from
the reference rather than the submission. Every response carries the reference
id, its scaler checksum, and the population hash of the cohort it was derived
from, so any score remains attributable to the population it is relative to.

Every response states which of the two it is, in `relative_to`.

## Consequences

The service cannot be used for one patient without someone first registering a
reference. That is a deliberate obstacle: registering a reference is the act of
deciding what the score is relative to, and it should be an explicit decision
rather than an implicit consequence of who else was in the batch.

The 20-sample floor for batch scoring is a judgement, not a derivation. Below
it the standardisation is dominated by noise; there is no threshold at which it
becomes sound, only one below which it is plainly unsound.

Two endpoints is more surface than one. The alternative — a single endpoint
that silently switched behaviour by sample count — would hide exactly the
distinction that matters.

## What would change this

A genuinely sample-independent scoring method (fixed reference ranges, an
absolute assay) would score one sample validly with no reference at all. None
of the published signatures here work that way, and `COHORT_RELATIVE_METHODS`
is asserted to cover every implemented method so that adding one forces a
deliberate decision.
