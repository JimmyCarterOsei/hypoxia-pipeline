"""Locked Phase 9C.2 molecular development; exports diagnostics, never model bundles."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hypoxiapipe.ingest.cache import sha256_file
from hypoxiapipe.ingest.cohort import Cohort
from hypoxiapipe.ingest.store import load_cohort
from hypoxiapipe.modeling.evaluation import paired_bootstrap
from hypoxiapipe.modeling.train import (
    CVResult,
    FoldResult,
    ModelingError,
    fit_partition,
    molecular_data,
    nested_cv,
    plan_cv,
    select_parameters,
)
from hypoxiapipe.provenance.hashing import hash_json
from hypoxiapipe.signatures.registry import load_bundled

POPULATION = "sha256:d9e702e02ffafa2d0ba56b3c4a548cfd75e137012185bdd070901b168053c30a"
EXPRESSION = "sha256:040d379670a28147796d2c04284967c977a28422a0a9c2bd6977d5be0007842b"
CLINICAL = "sha256:f9bfa3ab554253c0b7a7cb00574f74d2bf1411decab2104ac1fb47bbd3329a8c"
MODEL_NAMES = {
    "smith20_cox": "smith20_cox",
    "coxnet": "smith20_gene_coxnet",
    "rsf": "smith20_gene_rsf",
}


def development_design() -> dict[str, Any]:
    """Return the prespecified design; no parameters depend on observed performance."""
    return {
        "outer_splits": 5,
        "inner_splits": 5,
        "repeats": 5,
        "seed": 902001,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 902099,
        "permutation_repeats": 5,
        "final_seed": 922001,
        "primary_metric": "equal-weight mean of 25 outer-fold Harrell concordances",
        "uncertainty": (
            "patient-cluster percentile bootstrap conditional on frozen outer predictions"
        ),
        "tie_breaking": "first candidate in listed order on exact mean-inner-C ties",
        "candidate_failures": "reject a candidate if any inner fold fails or warns on convergence",
        "selection": "mean inner-validation Harrell C across all five folds",
        "coxnet_ties": "scikit-survival CoxNet native Breslow approximation",
        "baseline_ties": "Efron",
        "secondary_metrics": "omitted: no existing validated IPCW/time-grid evaluation workflow",
        "candidates": {
            "smith20_cox": [{}],
            "coxnet": [
                {"alpha": a, "l1_ratio": ratio}
                for a in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
                for ratio in (0.5, 0.9)
            ],
            "rsf": [
                {"n_estimators": 200, "min_samples_leaf": leaf, "max_features": features}
                for leaf in (5, 15)
                for features in ("sqrt", 0.75)
            ],
        },
    }


def verify_development_cohort(cohort: Cohort) -> None:
    """Reject a different expression population or an endpoint inconsistent with its ledger."""
    if cohort.population_hash != POPULATION or cohort.expr_checksum != EXPRESSION:
        raise ModelingError("not the locked Phase 9B STAR expression population")
    clinical = cohort.clinical
    if (len(clinical), int(clinical.event_uncapped.sum()), int(clinical.event.sum())) != (
        483,
        101,
        97,
    ):
        raise ModelingError("locked BCR counts do not match 483/101/97")
    steps = cohort.provenance.to_dict()["steps"]
    source: dict[str, Any] = next((s for s in steps if s["action"] == "clinical"), {})
    if source.get("clinical_source") != "tcga_bcr" or source.get("checksum") != CLINICAL:
        raise ModelingError("clinical source is not the audited longitudinal BCR snapshot")
    patients = {p["patient"]: p for p in source["patients"]}
    for patient, row in clinical.iterrows():
        record = patients[str(patient)]
        if not record["included"] or record["selected_day"] != row.time_days:
            raise ModelingError(f"clinical ledger discrepancy for {patient}")
        if record["event_uncapped"] != row.event_uncapped or record["event"] != row.event:
            raise ModelingError(f"clinical event discrepancy for {patient}")
    if not np.allclose(
        clinical.time_months, np.minimum(clinical.time_days / 30.44, 60), atol=1e-12, rtol=0
    ):
        raise ModelingError("clinical month/censoring derivation changed")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_cv(result: CVResult, directory: Path) -> None:
    """Write complete outer predictions, search diagnostics and numerical fit summaries."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(f) for f in result.folds]).to_csv(
        directory / "outer_folds.csv", index=False
    )
    pd.DataFrame(result.predictions).to_csv(directory / "outer_predictions.csv", index=False)
    _write_json(directory / "training_diagnostics.json", result.training)
    _write_json(directory / "inner_search.json", result.tuning)
    if result.coefficients:
        coefficients = pd.DataFrame(result.coefficients)
        coefficients.to_csv(directory / "outer_coefficients.csv", index=False)
        stability = []
        for feature, group in coefficients.groupby("feature", sort=False):
            values = group.coefficient
            nonzero = values != 0
            count = int(nonzero.sum())
            positive, negative = int((values > 0).sum()), int((values < 0).sum())
            stability.append(
                {
                    "feature": feature,
                    "n_fits": len(values),
                    "n_nonzero": count,
                    "selection_frequency": count / len(values),
                    "positive_fits": positive,
                    "negative_fits": negative,
                    "sign_consistency_when_selected": max(positive, negative) / count
                    if count
                    else None,
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
        pd.DataFrame(stability).to_csv(directory / "coefficient_stability.csv", index=False)
    if result.importance:
        importance = pd.DataFrame(result.importance)
        importance.to_csv(directory / "permutation_importance.csv", index=False)
        folds = importance.groupby(["repeat", "fold", "gene"]).c_drop.mean().reset_index()
        folds.to_csv(directory / "importance_by_fold.csv", index=False)
        repeats = folds.groupby(["repeat", "gene"]).c_drop.mean().reset_index()
        repeats.to_csv(directory / "importance_by_repeat.csv", index=False)
        summary = folds.groupby("gene").c_drop.agg(["mean", "std", "median", "min", "max"])
        summary["fraction_folds_positive"] = folds.groupby("gene").c_drop.apply(
            lambda x: (x > 0).mean()
        )
        summary.sort_values("mean", ascending=False).to_csv(directory / "importance_summary.csv")
        ranks = repeats.pivot(index="gene", columns="repeat", values="c_drop")
        ranks.corr(method="spearman").to_csv(directory / "importance_repeat_rank_correlations.csv")


def run_development(cohort_directory: str | Path, output_directory: str | Path) -> None:
    """Run the precommitted comparison and only then fit provisional full-cohort diagnostics.

    The output directory must be empty. Configuration and all patient-level
    nested splits are written before the first model fit. A failed run remains
    visibly incomplete; no folds, candidates or predictions are silently dropped.
    """
    source = Path(cohort_directory)
    out = Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise ModelingError("development output directory must be empty; do not overwrite a run")
    cohort = load_cohort(source, verify=True)
    verify_development_cohort(cohort)
    signature = load_bundled("smith20")
    features, y = molecular_data(
        cohort.expr, cohort.clinical.time_months, cohort.clinical.event, tuple(signature.genes)
    )
    design = development_design()
    splits = plan_cv(
        list(features.columns),
        y["event"],
        design["outer_splits"],
        design["inner_splits"],
        design["repeats"],
        design["seed"],
    )
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "planned",
        "started_utc": datetime.now(UTC).isoformat(),
        "design": design,
        "design_hash": hash_json(design),
        "splits_hash": hash_json(splits),
        "population_hash": cohort.population_hash,
        "expression_checksum": cohort.expr_checksum,
        "clinical_source_checksum": CLINICAL,
        "n": 483,
        "events_uncapped": 101,
        "events": 97,
        "genes": list(signature.genes),
        "signature_checksum": signature.checksum,
        "input_files": {
            name: sha256_file(source / name)
            for name in ("expression.parquet", "clinical.parquet", "cohort.json")
        },
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scikit-survival")
        },
        "implementation": {
            name: sha256_file(Path(__file__).parent / name)
            for name in (
                "train.py",
                "preprocessing.py",
                "evaluation.py",
                "development.py",
                "rbridge.py",
                "r/survival_worker.R",
            )
        },
        "no_final_production_model": True,
    }
    _write_json(out / "design.json", design)
    _write_json(out / "splits.json", splits)
    _write_json(out / "provenance.json", provenance)
    predictions: dict[str, pd.DataFrame] = {}
    for kind, name in MODEL_NAMES.items():

        def progress(fold: FoldResult) -> None:
            print(f"{fold.estimator}: repeat {fold.repeat} fold {fold.fold} complete", flush=True)

        result = nested_cv(
            features,
            y["time"],
            y["event"],
            kind,
            design["outer_splits"],
            design["inner_splits"],
            seed=design["seed"],
            repeats=design["repeats"],
            genes=tuple(signature.genes),
            candidates=design["candidates"][kind],
            splits=splits,
            permutation_repeats=design["permutation_repeats"] if kind == "rsf" else 0,
            progress=progress,
        )
        export_cv(result, out / name)
        predictions[name] = pd.DataFrame(result.predictions)
    comparison, paired, folds = paired_bootstrap(
        predictions, design["bootstrap_draws"], design["bootstrap_seed"]
    )
    comparison.to_csv(out / "comparison.csv", index=False)
    paired.to_csv(out / "paired_differences.csv", index=False)
    folds.to_csv(out / "outer_concordance.csv", index=False)
    folds.groupby(["model", "repeat"]).c_index.mean().reset_index().to_csv(
        out / "repeat_concordance.csv", index=False
    )
    provenance["status"] = "outer_cv_complete"
    _write_json(out / "provenance.json", provenance)
    # This gate is reached only after all three complete nested-CV evaluations.
    ids = np.asarray(features.columns)
    inner = StratifiedKFold(design["inner_splits"], shuffle=True, random_state=design["final_seed"])
    full_inner = [
        {"train": ids[a].tolist(), "validation": ids[b].tolist()}
        for a, b in inner.split(ids, y["event"])
    ]
    _write_json(out / "provisional_inner_splits.json", full_inner)
    for kind, name in MODEL_NAMES.items():
        params, tuning = select_parameters(
            features, y, kind, design["candidates"][kind], full_inner, design["final_seed"]
        )
        fit = fit_partition(features, y, kind, params, design["final_seed"])
        directory = out / name
        _write_json(directory / "provisional_inner_search.json", tuning)
        _write_json(
            directory / "provisional_diagnostic.json",
            {
                "diagnostic_only": True,
                "production_designation": False,
                "params": params,
                "n": len(y),
                "warnings": list(fit.warnings),
                "backend_detail": getattr(fit.model, "detail", None),
                "coefficients": (
                    np.asarray(fit.model.coef_).ravel().tolist()
                    if hasattr(fit.model, "coef_")
                    else None
                ),
                "coefficient_features": (
                    ["fixed_smith20_score"] if kind == "smith20_cox" else list(signature.genes)
                )
                if kind != "rsf"
                else None,
            },
        )
        pd.DataFrame(
            {"patient": features.columns, "risk": fit.predict(features).to_numpy()}
        ).to_csv(directory / "provisional_apparent_predictions.csv", index=False)
        print(f"{name}: provisional diagnostic fit complete", flush=True)
    provenance["status"] = "complete"
    provenance["completed_utc"] = datetime.now(UTC).isoformat()
    provenance["output_checksums"] = {
        str(p.relative_to(out)): sha256_file(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "provenance.json"
    }
    _write_json(out / "provenance.json", provenance)
