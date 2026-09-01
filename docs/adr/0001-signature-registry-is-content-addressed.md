# 0001. Signature specs are content-addressed and verified on load

## Status

Accepted.

## Context

The project began because four comparator signatures were wrong in a local
analysis, and none of the errors announced themselves. Three of four gene lists
were mislabelled. The Ragnum vector shared 4 of 32 genes with the published
signature, having been reconstructed from the gene-set categories rather than
the paper's Table 2. Buffa existed in two local variants, neither matching the
collaborator's. Every one of these produced a plausible hazard ratio.

The common feature is that a gene list is a plain data file that anyone can edit
and nothing checks. A signature that has drifted from its source looks exactly
like one that has not.

## Decision

A signature is a YAML spec carrying its source DOI, gene list, optional
published coefficients, pinned symbol authority, and a SHA-256 over its
canonical content excluding the checksum field. Loading verifies the hash and
raises `ChecksumMismatchError` on disagreement.

Re-hashing is a separate, explicit command (`sig hash --write`), so updating the
checksum is an act, not a side effect of editing.

## Consequences

Editing a gene list without re-verifying against source now breaks the build
loudly. That is the entire point, and it makes legitimate edits mildly annoying:
you must go back to the paper, confirm the change, and re-hash deliberately.

The checksum covers content, not correctness. A spec faithfully transcribing the
wrong table will hash cleanly forever. This guards against drift after
transcription, not against transcription error.

## What would change this

Nothing likely. If specs ever move into a database rather than files, the same
property has to survive the move.
