"""Resolve expression-matrix gene symbols to a pinned authority.

Alias drift is silent by construction: a signature gene listed as ``AK3L1``
simply fails to match a matrix indexed by ``AK4``, and the signature scores over
one gene fewer with no error anywhere. Coverage drops, the hazard ratio moves,
and nothing in the log says why.

This module makes each of those events explicit and countable:

* aliases remapped to their approved symbol,
* aliases that map to more than one approved symbol (ambiguous - never guessed),
* symbols not present in the authority (left as-is, but reported),
* duplicate rows created by remapping, collapsed under a stated rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hypoxiapipe.harmonise.aliases import AliasTable
from hypoxiapipe.signatures.registry import Signature

COLLAPSE_RULES = ("max_mean", "mean", "first")


@dataclass(frozen=True)
class SymbolReport:
    """What symbol harmonisation did to a matrix."""

    authority: str
    authority_checksum: str
    n_input: int
    n_output: int
    remapped: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    collapsed: dict[str, int] = field(default_factory=dict)
    collapse_rule: str = "max_mean"

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary for the QC report and run manifest."""
        return {
            "authority": self.authority,
            "authority_checksum": self.authority_checksum,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_remapped": len(self.remapped),
            "remapped": self.remapped,
            "n_ambiguous": len(self.ambiguous),
            "ambiguous": {k: list(v) for k, v in self.ambiguous.items()},
            "n_collapsed": len(self.collapsed),
            "collapsed": self.collapsed,
            "collapse_rule": self.collapse_rule,
        }


def collapse_duplicate_rows(
    expr: pd.DataFrame, rule: str = "max_mean"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse rows sharing a gene symbol under an explicit rule.

    ``max_mean`` keeps, for each symbol, the row with the highest mean
    expression - the usual microarray convention, chosen because averaging
    across probes with different hybridisation efficiency mixes scales.
    """
    if rule not in COLLAPSE_RULES:
        raise ValueError(f"unknown collapse rule {rule!r} (choose from {COLLAPSE_RULES})")
    dup_mask = expr.index.duplicated(keep=False)
    if not dup_mask.any():
        return expr, {}

    counts = {str(sym): int(n) for sym, n in expr.index[dup_mask].value_counts().items()}
    if rule == "mean":
        out = expr.groupby(level=0).mean()
    elif rule == "first":
        out = expr[~expr.index.duplicated(keep="first")]
    else:
        # Positional sort: label-based .loc on a duplicated index expands
        # rows combinatorially instead of reordering them.
        means = expr.mean(axis=1, skipna=True).to_numpy(dtype=float)
        order = np.argsort(-np.nan_to_num(means, nan=-np.inf), kind="stable")
        ranked = expr.iloc[order]
        winners = ranked[~ranked.index.duplicated(keep="first")]
        first_seen = list(dict.fromkeys(str(g) for g in expr.index))
        kept = {str(g) for g in winners.index}
        out = winners.loc[[g for g in first_seen if g in kept]]
    return out, counts


def harmonise_symbols(
    expr: pd.DataFrame,
    table: AliasTable,
    collapse_rule: str = "max_mean",
) -> tuple[pd.DataFrame, SymbolReport]:
    """Map matrix row labels onto approved symbols and collapse duplicates."""
    remapped: dict[str, str] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    new_index: list[str] = []

    for raw in expr.index:
        symbol = str(raw).strip()
        approved = table.approved_for(symbol)
        if len(approved) == 1 and approved[0] != symbol:
            remapped[symbol] = approved[0]
            new_index.append(approved[0])
        elif len(approved) > 1:
            # Never guess between competing targets - keep the original and report.
            ambiguous[symbol] = approved
            new_index.append(symbol)
        else:
            new_index.append(symbol)

    out = expr.copy()
    out.index = pd.Index(np.array(new_index, dtype=object), name=expr.index.name or "gene")
    out, collapsed = collapse_duplicate_rows(out, rule=collapse_rule)

    return out, SymbolReport(
        authority=table.authority,
        authority_checksum=table.checksum,
        n_input=int(expr.shape[0]),
        n_output=int(out.shape[0]),
        remapped=remapped,
        ambiguous=ambiguous,
        collapsed=collapsed,
        collapse_rule=collapse_rule,
    )


def check_signature_symbols(signature: Signature, table: AliasTable) -> dict[str, str]:
    """Report signature genes that are outdated under the pinned authority.

    Returned as ``{listed_symbol: approved_symbol}``. This is a warning, not an
    automatic fix: a signature's gene list is verified against its published
    source and hashed, so it is not silently rewritten. Harmonise the *matrix*
    towards the authority instead, or publish a new spec version deliberately.
    """
    out: dict[str, str] = {}
    for gene in signature.genes:
        approved = table.approved_for(gene)
        if len(approved) == 1 and approved[0] != gene:
            out[gene] = approved[0]
    return out
