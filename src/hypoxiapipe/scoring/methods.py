"""Signature scoring strategies.

All methods take an expression matrix with genes as rows and samples as
columns (Bioconductor convention) and return one score per sample.

Methods
-------
rowmean   : per-gene z-score across samples, then mean of z per sample.
            Equal gene weighting; robust to absolute expression scale.
median_z  : median of raw expression across signature genes per sample,
            then z across samples. Legacy method - dominated by
            high-expression genes; retained for method-sensitivity
            comparisons, not recommended as a primary score.
weighted  : sum(coefficient * per-gene z) per sample. For signatures
            published with Cox coefficients (directionally mixed gene
            sets cancel out under unweighted averaging).

Note on cohort relativity: per-gene standardisation makes every score
relative to the sample set being scored. Score the analysis population,
and record it - identical patients embedded in different cohorts get
different scores. This is inherent to z-based scoring, not a bug.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hypoxiapipe.errors import InsufficientGenesError, ScoringError
from hypoxiapipe.signatures.registry import Signature

DEFAULT_MIN_GENES = 3

_METHODS = ("rowmean", "median_z", "weighted")


@dataclass(frozen=True)
class ScoringResult:
    """Scores plus the provenance a downstream manifest needs."""

    scores: pd.Series
    signature: str
    checksum: str
    method: str
    n_found: int
    n_total: int
    missing: tuple[str, ...]

    @property
    def coverage(self) -> float:
        """Fraction of signature genes present in the matrix."""
        return self.n_found / self.n_total if self.n_total else 0.0


def _subset(matrix: pd.DataFrame, sig: Signature, min_genes: int) -> tuple[pd.DataFrame, list[str]]:
    present = [g for g in sig.genes if g in matrix.index]
    missing = [g for g in sig.genes if g not in matrix.index]
    if len(present) < min_genes:
        raise InsufficientGenesError(
            f"signature '{sig.name}': only {len(present)}/{sig.n_genes} genes "
            f"present in matrix (minimum {min_genes}). Missing: {missing[:10]}..."
        )
    sub = matrix.loc[present]
    if sub.shape[1] < 2:
        raise ScoringError("need at least 2 samples to standardise per gene")
    return sub, missing


def _zscore_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (row) across samples. Zero-variance genes -> NaN row."""
    mu = sub.mean(axis=1)
    sd = sub.std(axis=1, ddof=1)
    sd = sd.replace(0.0, np.nan)
    return sub.sub(mu, axis=0).div(sd, axis=0)


def score(
    matrix: pd.DataFrame,
    signature: Signature,
    method: str | None = None,
    min_genes: int = DEFAULT_MIN_GENES,
) -> ScoringResult:
    """Score every sample in `matrix` against `signature`.

    Parameters
    ----------
    matrix : pd.DataFrame
        Genes x samples expression matrix on a continuous scale
        (log / VST / array intensity).
    signature : Signature
        Verified signature from the registry.
    method : str | None
        Override the signature's default scoring method.
    min_genes : int
        Minimum number of signature genes that must be present.

    """
    method = method or signature.scoring
    if method not in _METHODS:
        raise ScoringError(f"unknown scoring method '{method}' (choose from {_METHODS})")
    if method == "weighted" and not signature.weighted:
        raise ScoringError(
            f"signature '{signature.name}' has no coefficients; weighted scoring is not applicable"
        )

    sub, missing = _subset(matrix, signature, min_genes)

    if method == "median_z":
        raw = sub.median(axis=0)
        vals = (raw - raw.mean()) / raw.std(ddof=1)
    else:
        z = _zscore_rows(sub).dropna(axis=0, how="all")
        if method == "rowmean":
            vals = z.mean(axis=0)
        else:  # weighted
            assert signature.coefficients is not None
            coef = pd.Series(signature.coefficients).reindex(z.index)
            vals = z.mul(coef, axis=0).sum(axis=0, skipna=True)

    scores = pd.Series(np.asarray(vals, dtype=float), index=sub.columns, name=signature.name)
    return ScoringResult(
        scores=scores,
        signature=signature.name,
        checksum=signature.checksum,
        method=method,
        n_found=sub.shape[0],
        n_total=signature.n_genes,
        missing=tuple(missing),
    )
