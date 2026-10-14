"""Data-only, checksum-verified STAR Smith20 Cox research model bundle.

Inference consumes frozen training statistics. It never fits or loads executable
serialized objects, and it does not estimate absolute recurrence probabilities.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.provenance.hashing import hash_file, hash_json
from hypoxiapipe.signatures.registry import load_bundled

MODEL_ID = "prad-bcr-star-smith20-cox-v1"
MODEL_VERSION = "1.0.0"
EXPRESSION_TYPE = "gdc_star_tpm_log2p1_hgnc"
EXTERNAL_INTERPRETATION = (
    "external validation of cohort-relative Smith20 association, "
    "not external validation of the frozen STAR-native predictor"
)
FILES = (
    "model.json",
    "scaler.json",
    "features.json",
    "training.json",
    "metrics.json",
    "provenance.json",
    "environment.json",
)


class BundleError(HypoxiapipeError):
    """A model bundle or inference request violates the locked contract."""


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise BundleError(f"invalid model JSON: {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"model JSON must be an object: {path.name}")
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _number(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BundleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise BundleError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


@dataclass(frozen=True)
class PredictionResult:
    """One or more patient predictions with explicit model provenance."""

    model_id: str
    model_version: str
    predictions: pd.DataFrame
    genes_used: tuple[str, ...]
    input_contract: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Smith20CoxBundle:
    """Frozen numerical representation of the research Cox model."""

    path: Path
    model: dict[str, Any]
    scaler: dict[str, Any]
    features: dict[str, Any]
    training: dict[str, Any]
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def model_id(self) -> str:
        """Stable research model identifier."""
        return MODEL_ID

    @property
    def model_version(self) -> str:
        """Stable semantic version."""
        return MODEL_VERSION

    def predict(
        self, expression: pd.DataFrame, *, expression_type: str | None = None
    ) -> PredictionResult:
        """Predict from HGNC genes x samples on declared log2(STAR TPM + 1) scale."""
        declared = expression_type or expression.attrs.get("expression_type")
        if declared != EXPRESSION_TYPE:
            raise BundleError(f"incompatible or undeclared expression type: {declared!r}")
        if not expression.index.is_unique:
            raise BundleError("duplicated genes after harmonisation")
        if not expression.columns.is_unique or not all(
            isinstance(sample, str) for sample in expression.columns
        ):
            raise BundleError("sample identifiers must be unique strings")
        if not len(expression.columns):
            raise BundleError("expression must contain at least one sample")
        genes = tuple(self.features["genes"])
        missing = [gene for gene in genes if gene not in expression.index]
        if missing:
            raise BundleError(f"missing Smith20 genes: {missing}")
        try:
            values = expression.loc[list(genes)].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise BundleError("required expression values must be numeric") from exc
        if not np.isfinite(values).all():
            raise BundleError("required expression values must be finite")
        mean = np.array([self.scaler["gene_means"][gene] for gene in genes])
        sd = np.array([self.scaler["gene_sample_sds"][gene] for gene in genes])
        raw = ((values - mean[:, None]) / sd[:, None]).mean(axis=0)
        score = (raw - self.scaler["composite_mean"]) / self.scaler["composite_sample_sd"]
        linear = score * self.model["coefficient"]
        with np.errstate(over="ignore", invalid="ignore"):
            hazard = np.exp(linear)
        if (
            not np.isfinite(score).all()
            or not np.isfinite(linear).all()
            or not np.isfinite(hazard).all()
        ):
            raise BundleError("nonfinite score or relative hazard")
        reference = np.asarray(self.training["score_reference_sorted"], dtype=float)
        percentile = 100 * np.searchsorted(reference, score, side="right") / len(reference)
        predictions = pd.DataFrame(
            {
                "smith20_score": score,
                "linear_predictor": linear,
                "relative_hazard": hazard,
                "training_percentile": percentile,
            },
            index=pd.Index(expression.columns, name="sample"),
        )
        return PredictionResult(
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            predictions=predictions,
            genes_used=genes,
            input_contract=self.features["input_contract"],
            provenance={
                "population_hash": self.training["population_hash"],
                "scaler_checksum": self.scaler["checksum"],
                "manifest_checksum": hash_json(self.manifest),
            },
        )


def load_model_bundle(path: str | Path) -> Smith20CoxBundle:
    """Load only verified JSON files and enforce cross-file scientific invariants."""
    root = Path(path)
    manifest = _read(root / "manifest.json")
    if manifest.get("model_id") != MODEL_ID or manifest.get("version") != MODEL_VERSION:
        raise BundleError("unexpected model identity or version")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(FILES):
        raise BundleError("manifest must list exactly the required bundle files")
    for name in FILES:
        try:
            observed = hash_file(root / name)
        except OSError as exc:
            raise BundleError(f"missing or unreadable bundle file: {name}: {exc}") from exc
        if hashes[name] != observed:
            raise BundleError(f"manifest checksum mismatch: {name}")
    model, scaler, features, training, metrics, provenance, _environment = (
        _read(root / name) for name in FILES
    )
    genes = tuple(load_bundled("smith20").genes)
    if (
        model.get("model_id") != MODEL_ID
        or model.get("version") != MODEL_VERSION
        or model.get("estimator") != "fixed_smith20_cox"
        or model.get("tie_method") != "efron"
        or model.get("status") != "research_use_only"
    ):
        raise BundleError("model definition does not match STAR Smith20 Cox v1")
    if (
        features.get("genes") != list(genes)
        or features.get("signature_checksum") != load_bundled("smith20").checksum
    ):
        raise BundleError("feature ordering or signature checksum mismatch")
    if features.get("expression_type") != EXPRESSION_TYPE:
        raise BundleError("incompatible bundle expression domain")
    if (
        features.get("missing_gene_policy") != "error"
        or features.get("extra_gene_policy") != "ignore"
    ):
        raise BundleError("unexpected gene handling policy")
    expected_scaler = hash_json({k: v for k, v in scaler.items() if k != "checksum"})
    if scaler.get("checksum") != expected_scaler or model.get("scaler_checksum") != expected_scaler:
        raise BundleError("model/scaler checksum mismatch")
    if model.get("features_checksum") != hash_json(features):
        raise BundleError("model/features checksum mismatch")
    if (
        scaler.get("genes") != list(genes)
        or set(scaler.get("gene_means", {})) != set(genes)
        or set(scaler.get("gene_sample_sds", {})) != set(genes)
    ):
        raise BundleError("scaler gene contract mismatch")
    for gene in genes:
        _number(scaler["gene_means"][gene], f"mean {gene}")
        _number(scaler["gene_sample_sds"][gene], f"SD {gene}", positive=True)
    _number(scaler.get("composite_mean"), "composite mean")
    _number(scaler.get("composite_sample_sd"), "composite SD", positive=True)
    _number(model.get("coefficient"), "Cox coefficient")
    if (
        scaler.get("n_train") != 483
        or model.get("training_n") != 483
        or model.get("training_events") != 97
        or scaler.get("population_hash") != training.get("population_hash")
        or training.get("n") != 483
        or training.get("events") != 97
    ):
        raise BundleError("training counts or population hash mismatch")
    reference = training.get("score_reference_sorted")
    if (
        not isinstance(reference, list)
        or len(reference) != 483
        or any(not math.isfinite(float(v)) for v in reference)
        or reference != sorted(reference)
    ):
        raise BundleError("invalid training percentile reference")
    evidence = metrics.get("external_signature_evidence", {})
    if (
        evidence.get("interpretation") != EXTERNAL_INTERPRETATION
        or metrics.get("frozen_predictor_external_validation") is not False
        or set(evidence.get("cohorts", {})) != {"MSKCC", "Cambridge", "Stockholm"}
    ):
        raise BundleError("external validation metadata misstates the evidence")
    if model.get("absolute_recurrence_probability_supported") is not False:
        raise BundleError("absolute recurrence probabilities are not supported")
    return Smith20CoxBundle(root, model, scaler, features, training, metrics, provenance, manifest)


def create_model_bundle(
    cohort_dir: str | Path,
    comparison_csv: str | Path,
    external_csv: str | Path,
    output_dir: str | Path,
) -> Smith20CoxBundle:
    """Refit M0 on the verified STAR/BCR cohort and write a transparent bundle."""
    from hypoxiapipe.ingest.store import load_cohort
    from hypoxiapipe.modeling.development import verify_development_cohort
    from hypoxiapipe.modeling.train import fit_partition, molecular_data

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise BundleError("bundle output directory must be empty")
    cohort = load_cohort(cohort_dir, verify=True)
    verify_development_cohort(cohort)
    signature = load_bundled("smith20")
    matrix, outcome = molecular_data(
        cohort.expr,
        cohort.clinical.time_months,
        cohort.clinical.event,
        tuple(signature.genes),
    )
    fitted = fit_partition(matrix, outcome, "smith20_cox", {}, seed=902001)
    if fitted.warnings:
        raise BundleError(f"full-cohort fit warnings: {fitted.warnings}")
    coefficient = _number(float(fitted.model.coef_[0]), "Cox coefficient")
    score = (
        fitted.scaler.transform(matrix).T.mean(axis=1) - fitted.composite_mean
    ) / fitted.composite_sd
    if not np.isfinite(score).all():
        raise BundleError("nonfinite full-cohort Smith20 scores")
    cv = pd.read_csv(comparison_csv).set_index("model")
    if "smith20_cox" not in cv.index or int(cast(Any, cv.loc["smith20_cox", "n_outer_fits"])) != 25:
        raise BundleError("missing 25-fit fixed Smith20 CV evidence")
    row = cv.loc[["smith20_cox"]].iloc[0]
    ext = pd.read_csv(external_csv).set_index("cohort")
    if (
        set(ext.index) != {"MSKCC", "Cambridge", "Stockholm"}
        or not ext.reproduced.all()
        or not (ext.coverage == "20/20").all()
    ):
        raise BundleError("external Smith20 association evidence is incomplete")
    domain = {
        "tissue": "prostate tumour",
        "assay": "GDC-compatible STAR TPM",
        "transform": "log2(TPM + 1)",
        "symbols": "hypoxiapipe HGNC harmonisation",
        "expression_type": EXPRESSION_TYPE,
        "unsupported": ["DESeq2 VST", "RSEM", "microarray", "FPKM", "arbitrary normalized matrix"],
    }
    features = {
        "genes": list(signature.genes),
        "signature_checksum": signature.checksum,
        "expression_type": EXPRESSION_TYPE,
        "input_contract": domain,
        "missing_gene_policy": "error",
        "extra_gene_policy": "ignore",
        "duplicate_gene_policy": "error",
    }
    if fitted.scaler.means is None or fitted.scaler.sds is None:
        raise BundleError("full-cohort gene scaler is unfitted")
    scaler = {
        "genes": list(signature.genes),
        "gene_means": {g: float(fitted.scaler.means[g]) for g in signature.genes},
        "gene_sample_sds": {g: float(fitted.scaler.sds[g]) for g in signature.genes},
        "composite_mean": fitted.composite_mean,
        "composite_sample_sd": fitted.composite_sd,
        "n_train": 483,
        "population_hash": cohort.population_hash,
        "sd_definition": "sample SD, ddof=1",
    }
    scaler["checksum"] = hash_json(scaler)
    model = {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "research_use_only",
        "estimator": "fixed_smith20_cox",
        "coefficient": coefficient,
        "tie_method": "efron",
        "endpoint": "longitudinal TCGA-PRAD biochemical recurrence",
        "censoring_horizon_months": 60,
        "training_n": 483,
        "training_events": 97,
        "scaler_checksum": scaler["checksum"],
        "features_checksum": hash_json(features),
        "absolute_recurrence_probability_supported": False,
        "outputs": ["smith20_score", "linear_predictor", "relative_hazard", "training_percentile"],
    }
    full = fitted.model.detail
    metrics = {
        "internal_repeated_nested_cv": {
            "metric": "held-out Harrell C",
            "folds": 25,
            "mean": float(cast(Any, row["mean_c"])),
            "sd": float(cast(Any, row["sd_c"])),
            "median": float(cast(Any, row["median_c"])),
            "bootstrap_ci": [
                float(cast(Any, row["ci_low"])),
                float(cast(Any, row["ci_high"])),
            ],
            "uncertainty": "patient-cluster bootstrap conditional on frozen outer predictions",
        },
        "full_cohort_apparent_fit": {
            "coefficient": coefficient,
            "se": float(full["se"]),
            "hr": float(full["hr"]),
            "ci_low": float(full["ci_low"]),
            "ci_high": float(full["ci_high"]),
            "wald_p": float(full["p"]),
            "fitted_c": float(full["c_index"]),
            "interpretation": "apparent fit; not held-out predictive performance",
        },
        "external_signature_evidence": {
            "interpretation": EXTERNAL_INTERPRETATION,
            "cohorts": {
                name: {
                    "n": int(cast(Any, ext.loc[name, "n"])),
                    "events": int(cast(Any, ext.loc[name, "events"])),
                    "hr": float(cast(Any, ext.loc[name, "HR"])),
                    "fitted_c": float(cast(Any, ext.loc[name, "Cindex"])),
                }
                for name in ("MSKCC", "Cambridge", "Stockholm")
            },
        },
        "frozen_predictor_external_validation": False,
    }
    training = {
        "n": 483,
        "events": 97,
        "population_hash": cohort.population_hash,
        "expression_checksum": cohort.expr_checksum,
        "score_reference_sorted": sorted(float(value) for value in score),
        "percentile_definition": "100 * count(training scores <= score) / 483",
    }
    provenance = {
        "clinical_source": "audited longitudinal TCGA-PRAD BCR",
        "cohort_manifest_checksum": hash_file(Path(cohort_dir) / "cohort.json"),
        "clinical_table_checksum": hash_file(Path(cohort_dir) / "clinical.parquet"),
        "expression_table_checksum": hash_file(Path(cohort_dir) / "expression.parquet"),
        "cv_comparison_checksum": hash_file(comparison_csv),
        "external_reproduction_checksum": hash_file(external_csv),
        "signature_checksum": signature.checksum,
        "scaler_checksum": scaler["checksum"],
    }
    environment = {
        "python_package": version("hypoxiapipe"),
        "r_version": full["r_version"],
        "survival_version": full["survival_version"],
        "representation": "checksum-verified JSON only; no pickle/joblib",
    }
    destination.mkdir(parents=True, exist_ok=True)
    payloads: tuple[dict[str, Any], ...] = (
        model,
        scaler,
        features,
        training,
        metrics,
        provenance,
        environment,
    )
    for name, payload in zip(FILES, payloads, strict=True):
        _write(destination / name, payload)
    _write(
        destination / "manifest.json",
        {
            "model_id": MODEL_ID,
            "version": MODEL_VERSION,
            "files": {name: hash_file(destination / name) for name in FILES},
        },
    )
    return load_model_bundle(destination)
