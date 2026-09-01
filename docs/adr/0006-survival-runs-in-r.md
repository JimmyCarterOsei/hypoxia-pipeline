# 0006. Survival estimation runs in R behind a JSON contract, with no Python fallback

## Status

Accepted.

## Context

The pipeline is Python. Survival estimation could be done in Python
(`lifelines`, `scikit-survival`) or delegated to R's `survival`, which is the
reference implementation this field's published results were produced with.

Reimplementing `survival` would cost weeks and be worse than the original.
Depending on a Python package instead is cheaper but means results are not
directly comparable to the R-produced numbers in the literature — and the
conventions that differ (tie handling, `ddof`, censoring at boundaries) are
exactly the ones that move a hazard ratio in the third decimal.

## Decision

Cox estimation runs in R's `survival`, reached over a deliberately narrow
contract: one JSON request on stdin, one JSON response on stdout, no shared
state and no file passing. The worker puts nothing else on stdout ever, because
a stray `cat()` would corrupt the response.

When R is unavailable, the bridge **raises**. There is no Python fallback.

Python and R agreement on the concordance index is asserted in CI.

## Consequences

R is a hard dependency of the modelling stage. That is a real deployment cost,
paid for by the containerised R stage (ADR 0010).

The no-fallback rule is the load-bearing part. A silent fallback would mean two
different estimators producing results under the same label, which is precisely
the failure this project came out of — an input quietly not being what it
claimed. An error is a worse user experience and a better guarantee.

The narrow contract turned out to have a second payoff: because the transport is
just stdin and stdout, it can be swapped for `docker run -i` with no code
change (ADR 0010).

## What would change this

Nothing foreseeable. If a Python implementation were ever adopted it would be an
additional named estimator with its own label, never a fallback for this one.
