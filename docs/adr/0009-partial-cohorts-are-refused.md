# 0009. A cohort missing files fails rather than shrinking

## Status

Accepted.

## Context

A TCGA cohort is assembled from several hundred per-aliquot files. Some fraction
of any bulk download fails: a timeout, a 500, a truncated response. The
convenient behaviour is to log the failures and build the matrix from what
arrived.

The problem is that the result is a different cohort from the one the manifest
describes, and the difference is invisible once the matrix exists. A cohort of
487 patients where 498 were requested looks exactly like a cohort of 487
patients. Every downstream number is computed on a population nobody chose.

## Decision

`tolerate_file_failures` defaults to 0. Any file that cannot be fetched or
parsed aborts the build, listing what failed and why. Tolerating losses is
possible but must be asked for, and the tolerated count is recorded in the
report and the manifest.

Cohort specs additionally pin `expect.n_samples`, asserted at build time, so a
re-versioned GDC release changing the file set fails rather than shifting every
result slightly.

## Consequences

Transient network failures fail whole builds. With a content-addressed cache
this is cheap — a re-run refetches only what is missing — but it is real
friction on a flaky connection.

Sample counts must be pinned deliberately after a first verified build, which
means an unpinned spec is honestly marked as unverified rather than appearing
checked.

## What would change this

Nothing about the default. A retry-with-backoff inside the fetcher would reduce
the friction without weakening the guarantee, and is the right fix if this
becomes annoying in practice.
