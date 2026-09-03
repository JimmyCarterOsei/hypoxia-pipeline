# Architecture decision records

Each record states a decision that was genuinely open at the time, the reasoning
that closed it, what it costs, and what would reopen it. They were written as the
decisions were made, not reconstructed afterwards — retrospective ADRs read as
invented because they are, and because the reasoning that actually drove a choice
is not recoverable a month later.

Records are immutable once accepted. A changed mind is a new record superseding
the old one, so the trail shows what was believed and when.

| # | Decision | Status |
|---|---|---|
| [0001](0001-signature-registry-is-content-addressed.md) | Signature specs are content-addressed and verified on load | Accepted |
| [0002](0002-scoring-method-is-required.md) | Scoring method is required, with no default | Accepted |
| [0003](0003-symbols-harmonised-before-scoring.md) | Symbols are harmonised against a pinned HGNC release before scoring | Accepted |
| [0004](0004-multi-target-probes-are-dropped.md) | Multi-target probes are dropped, not resolved to their first gene | Accepted |
| [0005](0005-analysis-set-precedes-standardisation.md) | Cohorts are restricted to their analysis set before any standardisation | Accepted |
| [0006](0006-survival-runs-in-r.md) | Survival estimation runs in R behind a JSON contract, with no Python fallback | Accepted |
| [0007](0007-preprocessing-frozen-inside-folds.md) | Preprocessing is fitted inside every fold and frozen | Accepted |
| [0008](0008-prad-endpoint-is-cdr-pfi.md) | TCGA-PRAD uses CDR PFI; overall survival is refused | Accepted |
| [0009](0009-partial-cohorts-are-refused.md) | A cohort missing files fails rather than shrinking | Accepted |
| [0010](0010-split-python-and-r-images.md) | Python and R ship as separate images | Accepted |
| [0011](0011-duckdb-deferred.md) | The DuckDB layer is deferred until something queries it | Accepted |
| [0012](0012-implausible-counts-are-checked.md) | Counts are checked for plausibility, not only for consistency | Accepted |
| [0013](0013-single-sample-scoring-is-refused.md) | Single-sample scoring is refused, not approximated | Accepted |

## Format

    # NNNN. Title
    ## Status
    ## Context      — what was true, and what was actually in tension
    ## Decision     — what was chosen
    ## Consequences — what it costs, not only what it buys
    ## What would change this
