# 0003. Symbols are harmonised against a pinned HGNC release before scoring

## Status

Accepted.

## Context

Gene symbols are renamed. `AK3L1` became `AK4`, `C3orf28` became `FAM162A`,
`CYR61` became `CCN1`. A signature published under the old name scores nothing
against a matrix labelled with the new one — and the failure mode is not an
error but a coverage report saying the gene is missing, followed by a score
computed over fewer genes than intended.

Resolving aliases against "current" HGNC is not reproducible either: the answer
changes when HGNC updates.

## Decision

Symbol harmonisation is a pipeline stage against a **pinned** HGNC release,
whose mapping table is itself hashed and recorded in the run manifest. The
release is named in the cohort spec. Remapped and dropped symbols are counted
and reported.

Harmonisation runs before scoring, always, and the ordering is enforced in
`build_cohort` rather than left to the caller.

## Consequences

The vendored table is a curated subset, not all of HGNC, so an unusual alias may
go unmapped. It is counted rather than silently passed through, so the loss is
visible.

Pinning means the pipeline uses deliberately out-of-date nomenclature between
updates. That is the correct trade: a result that changes because HGNC published
is not a reproducible result.

## What would change this

Moving to the full HGNC table with a pinned release date, if the curated subset
starts missing real aliases in practice.
