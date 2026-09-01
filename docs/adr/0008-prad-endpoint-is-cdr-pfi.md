# 0008. TCGA-PRAD uses CDR PFI; overall survival is refused

## Status

Accepted.

## Context

TCGA-PRAD has roughly 500 patients and about a dozen deaths during follow-up.
Prostate cancer patients rarely die within the study window, so overall survival
is not a conservative endpoint for this cohort — it is an empty one. A Cox model
fitted on it returns a hazard ratio, a confidence interval and a p-value that
all have the shape of results.

The GDC clinical endpoint carries no usable recurrence outcome for PRAD either.
The TCGA Clinical Data Resource (Liu et al., *Cell* 2018) curates four
harmonised endpoints and states which are usable per tumour type; for PRAD the
recommendation is PFI.

## Decision

TCGA endpoints come from the CDR table, and the bundled PRAD spec uses PFI.
`check_endpoint` **refuses** OS and DSS for PRAD by name, with the reason and
the recommended alternative in the error.

Overriding requires `allow_discouraged=True` in code — not a different string in
a config file. The easy path is the correct one; the wrong one must be said out
loud.

The CDR table is supplementary material from the paper and is not redistributed;
`clinical_path` points at the user's copy.

## Consequences

A dependency on a file the pipeline cannot fetch, with a clear error naming its
source. Worth it: the alternative is an endpoint that produces numbers nobody
should use.

The `DISCOURAGED` table covers only the projects this pipeline touches. An
unlisted project is unchecked rather than assumed fine, which is a gap that
grows if more projects are added.

## What would change this

Adding projects means extending `DISCOURAGED` from the CDR paper's own guidance
rather than leaving new projects silently unchecked.
