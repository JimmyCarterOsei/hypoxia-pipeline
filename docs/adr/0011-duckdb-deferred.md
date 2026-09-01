# 0011. The DuckDB layer is deferred until something queries it

## Status

Accepted.

## Context

The build plan includes DuckDB for run metadata and results, and it would be
straightforward to add: a schema, a writer, rows appended at the end of each
run.

It would also be, for now, a write-only sink. Nothing reads from it. The
reporting layer that would query it does not exist, and the per-run manifests
already carry the provenance that matters, verifiably.

A database that nothing queries reads as decoration on a portfolio project —
technology present to be listed rather than because a problem demanded it. The
plan flagged this specific risk before the layer was written.

## Decision

Not building it yet. Run metadata lives in per-run JSON manifests, which are
hashed and independently verifiable. DuckDB lands with the reporting layer that
reads from it, and the README says so explicitly rather than leaving the absence
to look like an oversight.

## Consequences

Cross-run queries currently mean reading manifests from disk. That is fine for
tens of runs and would not be for thousands — which is roughly the point at
which the layer earns its place.

Naming the deferral is itself the useful artefact: it shows a component was
considered and declined for a reason, which is more informative than one built
and unused.

## What would change this

A reporting or comparison layer that needs to query across runs. That is the
trigger, and it should arrive before the schema, not after.
