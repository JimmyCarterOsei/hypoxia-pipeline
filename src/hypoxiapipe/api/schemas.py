"""Request and response models for the scoring service.

The expression matrix is sent as genes x samples, matching the orientation used
everywhere else in the package. Sending it any other way is a silent
transposition bug waiting to happen, so the shape is validated on arrival
rather than trusted.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hypoxiapipe.signatures.registry import load_bundled

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
    service: str = "hypoxiapipe"
    version: str
    n_signatures: int
    n_references: int
    r_available: bool
    n_models: int = 0
    models_ready: bool = False
    model_registry_error: str | None = None
    git_commit: str | None = None
    image_tag: str | None = None
    base_image: str | None = None


class RecurrencePredictionRequest(BaseModel):
    """Research-use inference on an explicitly declared genes x samples matrix."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_id": "prad-bcr-star-smith20-cox-v1",
                    "expression_type": "gdc_star_tpm_log2p1_hgnc",
                    "matrix": {
                        "genes": list(load_bundled("smith20").genes),
                        "samples": ["patient-001"],
                        "values": [[1.0] for _ in load_bundled("smith20").genes],
                    },
                }
            ]
        }
    )

    model_id: str = Field(..., description="Registered model identifier, never a path or URL.")
    expression_type: str = Field(
        ...,
        description=(
            "Required for this model: gdc_star_tpm_log2p1_hgnc. "
            "No domain is inferred from expression values."
        ),
    )
    matrix: ExpressionMatrix = Field(
        ..., description="Gene-symbol rows x patient sample columns; all 20 Smith20 genes required."
    )


class RecurrencePrediction(BaseModel):
    """One sample's frozen-score and relative-hazard output."""

    sample_id: str
    smith20_score: float = Field(..., description="Standardized with frozen TCGA statistics.")
    linear_predictor: float = Field(..., description="Cox coefficient times Smith20 score.")
    relative_hazard: float = Field(..., description="exp(linear predictor), relative to score 0.")
    training_percentile: float = Field(
        ..., description="Rank in the TCGA development distribution."
    )


class RecurrenceProvenance(BaseModel):
    """Immutable model and training-source identifiers."""

    signature_checksum: str
    scaler_checksum: str
    population_hash: str
    manifest_checksum: str
    genes_used: list[str]


class RecurrenceInterpretation(BaseModel):
    """Limits on how the relative outputs may be read."""

    relative_hazard: str
    training_percentile: str


class RecurrencePredictionResponse(BaseModel):
    """Research-use prediction; no absolute recurrence probability is estimated."""

    model_id: str
    model_version: str
    endpoint: str = Field(..., description="Locked longitudinal BCR endpoint.")
    research_use_only: bool = Field(..., description="Always true for this unvalidated predictor.")
    expression_type: str = Field(..., description="Declared compatible STAR expression domain.")
    predictions: list[RecurrencePrediction]
    provenance: RecurrenceProvenance
    interpretation: RecurrenceInterpretation
    limitation: str = Field(..., description="Independent predictor-validation limitation.")


class ModelSummary(BaseModel):
    """Public summary of a configured, integrity-checked model."""

    model_id: str
    version: str
    estimator: str
    endpoint: str
    research_use_only: bool
    expected_expression_type: str
    n_features: int
    training_n: int
    training_events: int
    external_predictor_validation: bool


class ModelDetails(ModelSummary):
    """Safe full model metadata, without local source paths."""

    input_contract: dict[str, Any]
    feature_genes: list[str]
    internal_cv: dict[str, Any]
    censoring_horizon_months: float
    training_provenance: dict[str, str]
    external_signature_evidence: dict[str, Any]
    limitation: str
