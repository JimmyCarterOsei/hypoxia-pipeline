"""Strict file input and text representations for the frozen Smith20 Cox model."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hypoxiapipe.modeling.bundle import (
    EXPRESSION_TYPE,
    BundleError,
    PredictionResult,
    Smith20CoxBundle,
)

PREDICTION_COLUMNS = (
    "sample_id",
    "smith20_score",
    "linear_predictor",
    "relative_hazard",
    "training_percentile",
    "model_id",
    "model_version",
)
RESEARCH_WARNING = (
    "Research use only. relative_hazard is relative to the Cox model reference "
    "score of 0 and is not an absolute probability of recurrence."
)


def read_expression_matrix(path: Path, required_genes: tuple[str, ...]) -> pd.DataFrame:
    """Read genes x samples without letting pandas rename duplicate sample IDs."""
    if path.suffix.lower() not in {".tsv", ".csv"}:
        raise BundleError("expression input must be a .tsv or .csv matrix")
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream, delimiter=delimiter), [])
        if len(header) < 2:
            raise BundleError("expression matrix has zero sample columns; use genes x samples")
        samples = header[1:]
        if any(not name.strip() for name in samples):
            raise BundleError("sample identifiers cannot be blank")
        if len(samples) != len(set(samples)):
            raise BundleError("duplicate sample identifiers in expression header")
        if header[0].strip().lower() in {"sample", "sample_id"} or set(samples).intersection(
            required_genes
        ):
            raise BundleError("unsupported orientation: expected genes as rows, samples as columns")
        matrix = pd.read_csv(
            path, sep=delimiter, index_col=0, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise BundleError(f"cannot read expression matrix: {exc}") from exc
    if not matrix.columns.equals(pd.Index(samples)):
        raise BundleError("expression header changed during parsing")
    if matrix.index.hasnans or any(not str(gene).strip() for gene in matrix.index):
        raise BundleError("gene symbols cannot be blank")
    if not set(required_genes).intersection(matrix.index) and set(required_genes).intersection(
        matrix.columns
    ):
        raise BundleError("unsupported orientation: expected genes as rows, samples as columns")
    return matrix


def prediction_metadata(
    bundle: Smith20CoxBundle, result: PredictionResult, expression_type: str
) -> dict[str, Any]:
    """Return one provenance object shared by every prediction in a batch."""
    return {
        "model_id": result.model_id,
        "model_version": result.model_version,
        "status": bundle.model["status"],
        "signature_checksum": bundle.features["signature_checksum"],
        "scaler_checksum": result.provenance["scaler_checksum"],
        "population_hash": result.provenance["population_hash"],
        "expression_type": expression_type,
        "bundle_manifest_checksum": result.provenance["manifest_checksum"],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "genes_used": list(result.genes_used),
        "interpretation": RESEARCH_WARNING,
    }


def prediction_table(result: PredictionResult) -> pd.DataFrame:
    """Produce the stable seven-column public prediction schema."""
    table = result.predictions.reset_index().rename(columns={"sample": "sample_id"})
    table["model_id"] = result.model_id
    table["model_version"] = result.model_version
    return table.loc[:, list(PREDICTION_COLUMNS)]


def render_predictions(
    result: PredictionResult, metadata: dict[str, Any], output_format: str
) -> str:
    """Render table, TSV or JSON without inserting clinical claims."""
    table = prediction_table(result)
    if output_format == "json":
        return (
            json.dumps(
                {"metadata": metadata, "predictions": table.to_dict(orient="records")},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    if output_format == "tsv":
        return table.to_csv(sep="\t", index=False, float_format="%.17g")
    if output_format == "table":
        return table.to_string(index=False, float_format=lambda value: f"{value:.12g}") + "\n"
    raise BundleError("format must be table, json or tsv")


def inspect_metadata(bundle: Smith20CoxBundle) -> dict[str, Any]:
    """Summarize public model identity without disclosing local source paths."""
    cv = bundle.metrics["internal_repeated_nested_cv"]
    return {
        "model_id": bundle.model_id,
        "model_version": bundle.model_version,
        "status": bundle.model["status"],
        "endpoint": bundle.model["endpoint"],
        "censoring_horizon_months": bundle.model["censoring_horizon_months"],
        "training_n": bundle.model["training_n"],
        "training_events": bundle.model["training_events"],
        "feature_count": len(bundle.features["genes"]),
        "expression_type": EXPRESSION_TYPE,
        "internal_cv": {
            "metric": cv["metric"],
            "outer_fits": cv["folds"],
            "mean": cv["mean"],
            "sd": cv["sd"],
            "median": cv["median"],
            "bootstrap_ci": cv["bootstrap_ci"],
        },
        "external_validation": (
            "Cohort-relative Smith20 association reproduced externally; frozen "
            "STAR-native predictor lacks independent compatible external validation."
        ),
        "interpretation": RESEARCH_WARNING,
    }
