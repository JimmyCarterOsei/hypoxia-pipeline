# 0012. Counts are checked for plausibility, not only for consistency

## Status

Accepted.

## Context

The first external validation run produced a Cambridge cohort of 19 samples
with 19 events, and a Stockholm cohort of 45 with 40. Stockholm **passed QC
cleanly**. Both numbers were wrong.

GSE70768 and GSE70769 record `time_to_bcr_(months)` only for patients who
relapsed — a patient who never recurred has no time-to-recurrence, and their
follow-up sits in `total_follow_up_(months)` instead. Reading only the
event-time column silently discarded every censored patient at the analysis-set
step, which is a step that is *supposed* to drop samples lacking a usable
endpoint. Nothing malfunctioned. Every count reconciled with every other count.

The result looked like a small cohort rather than a broken one, and a small
cohort is unremarkable. It was caught by eye — the event count sitting flush
against the sample count — not by any check in the pipeline.

Existing guards did not apply. ADR 0009 asserts pinned sample counts, but these
specs were unpinned pending a first verified build. The QC layer checked whether
there were *enough* events, never whether there were implausibly many.

## Decision

Two changes.

`EndpointSpec` gains `censored_time_column`. Where an event did not occur and
the event-time is absent, follow-up time is used instead. It can never override
a real event time, and the number of observations taking it is reported.

QC fails any cohort whose event rate exceeds 85%. Real prognostic cohorts rarely
pass 60% even at long follow-up; above 85% a bug is far likelier than a finding.
The message names the probable cause and the field that fixes it.

The general principle: a count that reconciles with the pipeline's other counts
can still be impossible. Internal consistency is not plausibility, and only the
latter would have caught this.

## Consequences

An 85% threshold will eventually fire on a legitimate cohort — a
long-follow-up study of high-risk disease could exceed it. That is the right
direction to be wrong in: a false alarm costs one investigation, while the
silent version puts a number in a table.

Cohorts whose event-time and follow-up columns differ now need one more field
declared, which cannot be inferred and must come from inspecting the source.

## What would change this

A cohort that legitimately trips the threshold would justify making it
configurable per spec — with the value recorded, not defaulted away.
