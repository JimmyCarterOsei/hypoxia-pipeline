"""Cohort container: expression matrix, clinical table, and provenance.

A ``Cohort`` binds an expression matrix (genes x samples) to a clinical table
(samples x variables) and refuses to let them drift apart. Every ingest, QC and
harmonisation step returns a new ``Cohort`` with an appended provenance step, so
the object carries its own history.

Why the analysis-set discipline matters
---------------------------------------
Per-gene z-scoring is *cohort-relative*: a sample's score depends on which other
samples are in the matrix. Standardising over a full matrix and then subsetting
to the patients with outcome data gives a different answer from subsetting first
and then standardising. In the upstream research project that difference showed
up as a hazard ratio of 1.767 versus 1.798 in the same cohort - small, but
unexplained until the cause was found.

``restrict_to_analysis_set()`` makes the decision explicit and records it, and
``population_hash`` identifies exactly which samples a score was computed over.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from hypoxiapipe.errors import CohortAlignmentError, EndpointError


def _hash_index(values: list[str]) -> str:
    """SHA-256 over a sorted list of identifiers."""
    digest = hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
    return f"sha256:{digest}"


def frame_checksum(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame's labels and values.

    Stable across row/column ordering so that a reordered but otherwise
    identical matrix hashes the same.
    """
    ordered = df.sort_index(axis=0).sort_index(axis=1)
    h = hashlib.sha256()
    h.update("\n".join(map(str, ordered.index)).encode())
    h.update(b"\x00")
    h.update("\n".join(map(str, ordered.columns)).encode())
    h.update(b"\x00")
    hashed = pd.util.hash_pandas_object(ordered.stack(future_stack=True), index=True)
    h.update(np.asarray(hashed).tobytes())
    return f"sha256:{h.hexdigest()}"


@dataclass(frozen=True)
class ProvenanceStep:
    """One recorded transformation applied to a cohort."""

    action: str
    detail: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


@dataclass(frozen=True)
class Provenance:
    """Where a cohort came from and what has been done to it."""

    source: str
    accession: str | None = None
    url: str | None = None
    platform: str | None = None
    retrieved_at: str | None = None
    symbol_authority: str | None = None
    steps: tuple[ProvenanceStep, ...] = ()

    def with_step(self, action: str, **detail: Any) -> Provenance:
        """Return a copy with one more step appended."""
        return replace(self, steps=(*self.steps, ProvenanceStep(action=action, detail=detail)))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, for the Phase 3 run manifest."""
        return {
            "source": self.source,
            "accession": self.accession,
            "url": self.url,
            "platform": self.platform,
            "retrieved_at": self.retrieved_at,
            "symbol_authority": self.symbol_authority,
            "steps": [{"action": s.action, "at": s.at, **s.detail} for s in self.steps],
        }


@dataclass(frozen=True)
class Cohort:
    """An aligned expression matrix and clinical table with provenance.

    Parameters
    ----------
    name : short cohort label used in reports (e.g. "TCGA-PRAD").
    expr : genes x samples, continuous scale (log / VST / array intensity).
    clinical : samples x variables, indexed by sample ID.
    provenance : where it came from and what has been done to it.

    """

    name: str
    expr: pd.DataFrame
    clinical: pd.DataFrame
    provenance: Provenance

    def __post_init__(self) -> None:
        """Validate that expression and clinical tables are aligned."""
        if self.expr.columns.duplicated().any():
            dupes = sorted(self.expr.columns[self.expr.columns.duplicated()].unique())
            raise CohortAlignmentError(f"{self.name}: duplicate sample IDs in expression: {dupes}")
        if self.clinical.index.duplicated().any():
            dupes = sorted(self.clinical.index[self.clinical.index.duplicated()].unique())
            raise CohortAlignmentError(f"{self.name}: duplicate sample IDs in clinical: {dupes}")
        if list(self.expr.columns) != list(self.clinical.index):
            only_expr = sorted(set(self.expr.columns) - set(self.clinical.index))[:5]
            only_clin = sorted(set(self.clinical.index) - set(self.expr.columns))[:5]
            raise CohortAlignmentError(
                f"{self.name}: expression and clinical samples are not aligned "
                f"(expression-only: {only_expr}, clinical-only: {only_clin}). "
                "Use Cohort.align() to take the intersection explicitly."
            )

    @property
    def n_genes(self) -> int:
        """Return the number of genes (rows)."""
        return self.expr.shape[0]

    @property
    def n_samples(self) -> int:
        """Return the number of samples (columns)."""
        return self.expr.shape[1]

    @property
    def population_hash(self) -> str:
        """Return a hash of the sample set - the population z-scores are relative to."""
        return _hash_index([str(c) for c in self.expr.columns])

    @property
    def expr_checksum(self) -> str:
        """Return the content hash of the expression matrix."""
        return frame_checksum(self.expr)

    @classmethod
    def align(
        cls,
        name: str,
        expr: pd.DataFrame,
        clinical: pd.DataFrame,
        provenance: Provenance,
        min_samples: int = 30,
    ) -> Cohort:
        """Build a cohort from the intersection of expression and clinical samples.

        Records how many samples each side contributed, so a silent 90% drop
        from an ID-format mismatch is visible rather than assumed.
        """
        common = [c for c in expr.columns if c in set(clinical.index)]
        if len(common) < min_samples:
            raise CohortAlignmentError(
                f"{name}: only {len(common)} samples align between expression "
                f"({expr.shape[1]}) and clinical ({clinical.shape[0]}); "
                f"minimum is {min_samples}. Check sample ID formats (e.g. TCGA "
                "barcode truncation) before proceeding."
            )
        prov = provenance.with_step(
            "align",
            n_expression=int(expr.shape[1]),
            n_clinical=int(clinical.shape[0]),
            n_aligned=len(common),
        )
        return cls(
            name=name, expr=expr.loc[:, common], clinical=clinical.loc[common], provenance=prov
        )

    def subset_samples(self, samples: list[str], reason: str) -> Cohort:
        """Return a cohort restricted to `samples`, recording why."""
        keep = [s for s in self.expr.columns if s in set(samples)]
        prov = self.provenance.with_step(
            "subset_samples", reason=reason, n_before=self.n_samples, n_after=len(keep)
        )
        return Cohort(
            name=self.name,
            expr=self.expr.loc[:, keep],
            clinical=self.clinical.loc[keep],
            provenance=prov,
        )

    def restrict_to_analysis_set(self, time_col: str, event_col: str) -> Cohort:
        """Drop samples without a usable survival endpoint, before any scoring.

        Scores are cohort-relative, so this must happen *before* standardisation,
        not after. Doing it in the other order changes every score slightly and
        the difference is invisible unless you look for it.
        """
        for col in (time_col, event_col):
            if col not in self.clinical.columns:
                raise EndpointError(
                    f"{self.name}: clinical table has no column '{col}' "
                    f"(available: {list(self.clinical.columns)[:10]})"
                )
        time = pd.to_numeric(self.clinical[time_col], errors="coerce")
        event = pd.to_numeric(self.clinical[event_col], errors="coerce")
        usable = time.notna() & event.notna() & (time > 0) & event.isin([0, 1])
        keep = [str(s) for s in self.clinical.index[usable]]
        if not keep:
            raise EndpointError(
                f"{self.name}: no samples have a usable endpoint in ('{time_col}', '{event_col}')"
            )
        out = self.subset_samples(
            keep, reason=f"analysis set for endpoint ({time_col}, {event_col})"
        )
        prov = out.provenance.with_step(
            "analysis_set",
            time_col=time_col,
            event_col=event_col,
            n_events=int(event[usable].sum()),
            population_hash=out.population_hash,
        )
        return Cohort(name=out.name, expr=out.expr, clinical=out.clinical, provenance=prov)

    def with_expression(self, expr: pd.DataFrame, action: str, **detail: Any) -> Cohort:
        """Return a cohort with a replaced expression matrix and a recorded step."""
        return Cohort(
            name=self.name,
            expr=expr.loc[:, list(self.expr.columns)],
            clinical=self.clinical,
            provenance=self.provenance.with_step(action, **detail),
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact description for logs and QC reports."""
        return {
            "name": self.name,
            "n_genes": self.n_genes,
            "n_samples": self.n_samples,
            "population_hash": self.population_hash,
            "expr_checksum": self.expr_checksum,
            "provenance": self.provenance.to_dict(),
        }
