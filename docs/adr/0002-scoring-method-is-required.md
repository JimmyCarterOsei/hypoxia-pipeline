# 0002. Scoring method is required, with no default

## Status

Accepted.

## Context

Yang was scored unweighted when it requires published Cox coefficients. Scored
unweighted it was null; scored with its coefficients it gave HR 1.63,
P = 4e-4 in TCGA-PRAD. Nothing errored in either case.

Assuming a scoring method is the same class of error as assuming a gene list:
both silently substitute a different analysis for the intended one. A default
would have made the null result look like a finding about the signature rather
than a fact about how it was scored.

## Decision

`scoring` is a required field in every signature spec. A spec without one raises
`IncompleteSpecError` at load. Weighted scoring on a signature with no
coefficients is an error rather than a silent fallback to unweighted.

The effect of scoring choice is reported as a first-class result — each
signature scored by its published method, with the sensitivity to that choice
quantified separately.

## Consequences

Every spec must state something the source paper sometimes leaves ambiguous.
Where a paper is genuinely unclear, that ambiguity now has to be resolved and
recorded rather than absorbed into a default.

## What would change this

Nothing. If anything, the same treatment should extend to other analysis choices
that currently have defaults.
