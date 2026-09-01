"""Preprocessing that is fitted once and then frozen.

Per-gene standardisation is the step where leakage gets in. Standardising a
matrix that contains the validation samples lets their distribution influence
the training features, and the resulting optimism is invisible: nothing errors,
the C-index is simply a little too high.

:class:`ReferenceScaler` therefore separates the two operations that are
normally one call. ``fit`` computes per-gene means and standard deviations from
a declared training population and records which samples it saw.
``transform`` applies exactly those constants to any other matrix. A fitted
scaler is a serialisable artefact: the same constants that trained the model
are what a later cohort - or the Phase 8 scoring service - is scored against.

This is the same cohort-relativity problem the ingest layer handles by pinning
the analysis set, met again one stage downstream. Scoring against frozen
reference statistics is what makes a single-sample prediction well defined at
all, which is why the service in the build plan exposes ``/score/reference``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.provenance.hashing import hash_json

MIN_SD = 1e-8


class PreprocessingError(HypoxiapipeError):
    """Preprocessing was misused - unfitted, refitted, or given unknown genes."""


@dataclass
class ReferenceScaler:
    """Per-gene standardisation with statistics frozen at fit time."""

    genes: tuple[str, ...] = ()
    means: pd.Series | None = None
    sds: pd.Series | None = None
    n_fit_samples: int = 0
    population_hash: str | None = None
    fitted: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    # -- fitting ----------------------------------------------------------
    def fit(self, matrix: pd.DataFrame, population_hash: str | None = None) -> ReferenceScaler:
        """Compute per-gene statistics from a training matrix (genes x samples).

        Refitting a fitted scaler is an error rather than a silent overwrite:
        in a nested CV loop it almost always means the outer fold's scaler has
        been reused where a new one was intended.
        """
        if self.fitted:
            raise PreprocessingError(
                "this scaler is already fitted; construct a new one rather than refitting "
                "(a refit inside a CV loop usually means a fold boundary was crossed)"
            )
        if matrix.shape[1] < 2:
            raise PreprocessingError("need at least 2 samples to compute a standard deviation")

        means = matrix.mean(axis=1)
        sds = matrix.std(axis=1, ddof=1)
        degenerate = [str(g) for g in sds.index[sds.fillna(0.0) < MIN_SD]]

        self.genes = tuple(str(g) for g in matrix.index)
        self.means = means
        self.sds = sds.mask(sds < MIN_SD)
        self.n_fit_samples = int(matrix.shape[1])
        self.population_hash = population_hash
        self.fitted = True
        self.detail = {
            "n_genes": len(self.genes),
            "n_fit_samples": self.n_fit_samples,
            "n_zero_variance": len(degenerate),
            "zero_variance_genes": degenerate[:20],
        }
        return self

    # -- applying ---------------------------------------------------------
    def transform(self, matrix: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        """Standardise a matrix using the frozen statistics.

        With ``strict=True`` a gene missing from the target matrix is an error,
        because a model expecting a feature it does not receive produces a
        number rather than a complaint.
        """
        if not self.fitted or self.means is None or self.sds is None:
            raise PreprocessingError("scaler is not fitted")

        missing = [g for g in self.genes if g not in matrix.index]
        if missing and strict:
            raise PreprocessingError(
                f"{len(missing)} of {len(self.genes)} fitted genes are absent from the "
                f"target matrix (first few: {missing[:8]}). Harmonise symbols first, or "
                "pass strict=False to score on the intersection and record the loss."
            )
        present = [g for g in self.genes if g in matrix.index]
        if not present:
            raise PreprocessingError("no fitted genes present in the target matrix")

        sub = matrix.loc[present]
        centred = sub.sub(self.means.loc[present], axis=0)
        scaled = centred.div(self.sds.loc[present], axis=0)
        # Zero-variance genes carry no information; a constant column is
        # preferable to inf, and the count is already in `detail`.
        return scaled.fillna(0.0)

    def fit_transform(
        self, matrix: pd.DataFrame, population_hash: str | None = None
    ) -> pd.DataFrame:
        """Fit on a matrix and transform it - valid only for training data."""
        return self.fit(matrix, population_hash=population_hash).transform(matrix)

    # -- provenance -------------------------------------------------------
    @property
    def checksum(self) -> str:
        """Content hash of the frozen statistics."""
        if not self.fitted or self.means is None or self.sds is None:
            raise PreprocessingError("scaler is not fitted")
        return hash_json(
            {
                "genes": list(self.genes),
                "means": [round(float(v), 10) for v in self.means],
                "sds": [None if pd.isna(v) else round(float(v), 10) for v in self.sds],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form of the fitted scaler."""
        if not self.fitted or self.means is None or self.sds is None:
            raise PreprocessingError("scaler is not fitted")
        return {
            "genes": list(self.genes),
            "means": {str(g): float(v) for g, v in self.means.items()},
            "sds": {str(g): (None if pd.isna(v) else float(v)) for g, v in self.sds.items()},
            "n_fit_samples": self.n_fit_samples,
            "population_hash": self.population_hash,
            "checksum": self.checksum,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReferenceScaler:
        """Rebuild a scaler from :meth:`to_dict` output, verifying its checksum."""
        genes = tuple(str(g) for g in raw["genes"])
        means = pd.Series({g: float(raw["means"][g]) for g in genes})
        sds = pd.Series(
            {g: (np.nan if raw["sds"][g] is None else float(raw["sds"][g])) for g in genes}
        )
        scaler = cls(
            genes=genes,
            means=means,
            sds=sds,
            n_fit_samples=int(raw.get("n_fit_samples", 0)),
            population_hash=raw.get("population_hash"),
            fitted=True,
            detail=raw.get("detail", {}),
        )
        recorded = raw.get("checksum")
        if recorded and recorded != scaler.checksum:
            raise PreprocessingError(
                "scaler checksum mismatch: the frozen statistics have changed since "
                "they were written. Refit deliberately rather than scoring against "
                "constants of unknown origin."
            )
        return scaler

    def save(self, path: str | Path) -> Path:
        """Write the fitted scaler to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> ReferenceScaler:
        """Load a fitted scaler from JSON, verifying its checksum."""
        return cls.from_dict(json.loads(Path(path).read_text()))
