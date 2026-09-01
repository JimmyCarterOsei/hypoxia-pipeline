"""Infer expression scale and assay family from the matrix itself.

Scoring assumes a continuous, roughly log-scaled matrix. Handing it raw counts
or linear TPM does not fail - it silently produces scores dominated by the few
highest-expressed genes. That is the same failure mode as the median-scoring
method this project replaced, arriving from a different direction, so the
pipeline checks rather than assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOG_MAX_PLAUSIBLE = 30.0
COUNT_INTEGER_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ScaleReport:
    """Inferred scale of an expression matrix."""

    scale: str  # "log", "linear", "z", "unknown"
    assay: str  # "rna-seq-counts", "rna-seq-normalised", "array", "unknown"
    minimum: float
    maximum: float
    median: float
    frac_negative: float
    integer_valued: bool
    recommendation: str | None
    #: True when the range above came from a subsample rather than every value.
    sampled: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary."""
        return {
            "scale": self.scale,
            "assay": self.assay,
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "range_is_sampled": self.sampled,
            "median": round(self.median, 4),
            "frac_negative": round(self.frac_negative, 4),
            "integer_valued": self.integer_valued,
            "recommendation": self.recommendation,
        }


def infer_scale(expr: pd.DataFrame, sample_n: int = 200_000) -> ScaleReport:
    """Infer whether a matrix is log-scaled, linear, or already standardised.

    Large matrices are subsampled to ``sample_n`` values, so ``minimum`` and
    ``maximum`` are the range *of the sample*, not of the matrix. That is fine
    for classifying the scale and misleading if read as an exact range, so the
    report labels them as sampled whenever subsampling occurred.
    """
    values = expr.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    sampled = values.size > sample_n
    if values.size == 0:
        return ScaleReport("unknown", "unknown", np.nan, np.nan, np.nan, np.nan, False, None, False)
    if values.size > sample_n:
        rng = np.random.default_rng(0)
        values = rng.choice(values, size=sample_n, replace=False)

    vmin, vmax = float(values.min()), float(values.max())
    vmed = float(np.median(values))
    frac_neg = float((values < 0).mean())
    integer_valued = bool(np.all(np.abs(values - np.round(values)) < COUNT_INTEGER_TOLERANCE))

    if integer_valued and vmax > LOG_MAX_PLAUSIBLE:
        scale, assay = "linear", "rna-seq-counts"
        rec = (
            "Raw counts detected. Normalise (VST/TPM) and log-transform before "
            "scoring; z-scoring raw counts weights genes by library-size artefact."
        )
    elif vmax > LOG_MAX_PLAUSIBLE:
        scale, assay = "linear", "rna-seq-normalised"
        rec = f"Linear scale detected (max {vmax:.1f}). Apply log2(x + 1) before scoring."
    elif frac_neg > 0.4 and abs(vmed) < 0.5 and vmax < 10:
        scale, assay = "z", "unknown"
        rec = (
            "Matrix appears already standardised per gene. Re-standardising is "
            "harmless but the population it was standardised over is unknown - "
            "record it, because scores are cohort-relative."
        )
    elif frac_neg > 0.01:
        scale, assay = "log", "array"
        rec = None
    else:
        scale, assay = "log", "rna-seq-normalised"
        rec = None

    return ScaleReport(
        scale=scale,
        assay=assay,
        minimum=vmin,
        maximum=vmax,
        median=vmed,
        frac_negative=frac_neg,
        integer_valued=integer_valued,
        recommendation=rec,
        sampled=sampled,
    )
