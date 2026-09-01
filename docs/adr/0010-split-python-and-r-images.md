# 0010. Python and R ship as separate images

## Status

Accepted.

## Context

The pipeline needs Python for everything except survival estimation, which needs
R (ADR 0006). One image containing both is simpler to run: no transport, no
indirection, one thing to build.

It is also larger, slower to rebuild, hides which stage depends on what, and —
decisively — makes Nextflow's per-process containers pointless, because every
process would run the same image. "One fat container" was on the project's own
list of traps before either image existed.

## Decision

Two images. `hypoxiapipe` holds ingest, QC, harmonisation, scoring, modelling
and provenance, and installs no R. `hypoxiapipe-r` holds R, `survival`,
`jsonlite` and the worker as its entrypoint, and installs no Python package.
Both run as non-root.

`HYPOXIAPIPE_R_COMMAND` replaces the command used to reach the worker:

    HYPOXIAPIPE_R_COMMAND="docker run --rm -i ghcr.io/OWNER/hypoxiapipe-r:TAG"

This works only because the Phase 4 contract is stdin and stdout (ADR 0006).
No service, no socket, no shared volume.

## Consequences

The survival call now crosses a process boundary that can fail in new ways — a
missing image, a daemon that is not running — so the bridge reports the command
it tried to run when startup fails.

The transport indirection is testable without a container daemon: any command
honouring the contract is a valid transport, so CI exercises the same code path
with a wrapper script, and the images are smoke-tested separately where a daemon
exists.

Two images means two build pipelines and two things to keep in sync.

## What would change this

If the R stage grew to need data files or state, the stdin/stdout contract would
strain and a service boundary might be warranted. Nothing suggests that yet.
