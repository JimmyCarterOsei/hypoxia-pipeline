# 0004. Multi-target probes are dropped, not resolved to their first gene

## Status

Accepted.

## Context

Array annotation encodes a probe matching several genes as `BNIP3 /// BNIP3P1`.
Something must be done with these. The common convention is to take the first
listed target, which is convenient and produces a fuller matrix.

But the first listed target is an artefact of how the annotation file was
written, not evidence about which gene the measurement reflects. Taking it
attributes one measurement to one arbitrarily chosen gene — and in the example
above, to a gene rather than its pseudogene.

## Decision

Multi-target probes are excluded by default and counted in the mapping report.
`multi="first"` exists for reproducing legacy analyses that did take the first,
but it must be asked for.

Unmapped probes (controls, retired identifiers) are likewise dropped and
counted rather than carried forward.

## Consequences

Matrices are smaller, and signature coverage can be lower than other tools
report for the same platform. When a signature's coverage looks disappointing,
the report says how many probes were lost and why, which makes the difference
explicable rather than mysterious.

Reproducing a published analysis that used the first-target convention requires
setting the flag — that is the intended friction.

## What would change this

If a signature gene were reachable only through multi-target probes on a given
platform, the honest options are to record the coverage loss or set the flag
deliberately for that cohort. Not to change the default.
