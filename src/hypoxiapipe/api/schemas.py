"""Request and response models for the scoring service.

The expression matrix is sent as genes x samples, matching the orientation used
everywhere else in the package. Sending it any other way is a silent
transposition bug waiting to happen, so the shape is validated on arrival
rather than trusted.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_GENES = 100_000
MAX_SAMPLES = 5_000


class ExpressionMatrix(BaseModel):
    """A genes x samples expression matrix on a continuous (log) scale."""

    genes: list[str] = Field(..., min_length=1, max_length=MAX_GENES)
    samples: list[str] = Field(..., min_length=1, max_length=MAX_SAMPLES)
    values: list[list[float]] = Field(..., description="One row per gene, one column per sample.")

    @field_validator("genes", "samples")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate identifiers are not allowed")
        return value

    @model_validator(mode="after")
    def _shape_matches(self) -> ExpressionMatrix:
        if len(self.values) != len(self.genes):
            raise ValueError(
                f"values has {len(self.values)} rows but {len(self.genes)} genes were named; "
                "the matrix must be genes x samples"
            )
        for i, row in enumerate(self.values):
            if len(row) != len(self.samples):
                raise ValueError(
                    f"row {i} has {len(row)} values but {len(self.samples)} samples were named"
                )
        return self

    def to_frame(self) -> pd.DataFrame:
        """Return the matrix as a DataFrame indexed by gene."""
        return pd.DataFrame(self.values, index=self.genes, columns=self.samples, dtype=float)

    @property
    def n_samples(self) -> int:
        """Number of samples in the request."""
        return len(self.samples)


class BatchScoreRequest(BaseModel):
    """Score a complete cohort, relative to itself."""

    signature: str = Field(..., description="Bundled signature name, e.g. 'smith20'.")
    matrix: ExpressionMatrix
    method: str | None = Field(
        None, description="Override the signature's published scoring method."
    )


class ReferenceScoreRequest(BaseModel):
    """Score samples against frozen statistics from a registered reference."""

    reference_id: str
    matrix: ExpressionMatrix
    strict: bool = Field(
        True,
        description=(
            "Require every reference gene to be present. With false, score on the "
            "intersection and report the missing genes."
        ),
    )


class ScoreResponse(BaseModel):
    """Scores plus the provenance needed to reproduce them."""

    scores: dict[str, float]
    signature: str
    signature_checksum: str
    method: str
    n_samples: int
    n_genes_used: int
    n_genes_expected: int
    missing_genes: list[str] = []
    relative_to: str = Field(
        ...,
        description=(
            "What the scores are relative to: 'submitted cohort' for batch scoring, "
            "or the reference id for reference scoring."
        ),
    )
    provenance: dict[str, Any] = {}


class ReferenceSummary(BaseModel):
    """One registered reference, as listed by the service."""

    reference_id: str
    signature: str
    signature_checksum: str
    method: str
    scaler_checksum: str
    cohort: str
    population_hash: str | None = None
    n_train: int
    n_genes: int
    created: str


class HealthResponse(BaseModel):
    """Liveness and configuration, for a load balancer or a curious human."""

    status: str
    version: str
    n_signatures: int
    n_references: int
    r_available: bool
