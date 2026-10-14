"""The scoring and frozen-model research inference service.

Two endpoints, and the difference between them is the whole point.

``/score/batch`` scores a complete cohort relative to itself. This is what the
published methods describe, and it is only valid on a cohort: per-gene
z-scoring standardises against the samples present, so the same patient scores
differently depending on who else is in the batch. The service therefore
**rejects** a single-sample batch request with 422 rather than returning a
number that means nothing.

``/score/reference`` scores against frozen statistics from a named cohort. This
is well defined for one sample, because the distribution comes from the
reference rather than from the submission. Every response says which of the two
it is, in ``relative_to``.

That rejection is the design, not a limitation to apologise for. An endpoint
that accepted one sample and returned a plausible number would be the most
dangerous thing in this repository: silently wrong, confidently formatted, and
indistinguishable from a real result. ``/predict/recurrence`` is different:
it applies a verified trained model with frozen development-cohort statistics.

Run it with::

    HYPOXIAPIPE_REFERENCES=references HYPOXIAPIPE_MODELS=models \
        uvicorn hypoxiapipe.api.app:app
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from hypoxiapipe import __version__
from hypoxiapipe.api.models import ModelRegistryError, list_models
from hypoxiapipe.api.references import (
    ReferenceError,
    list_references,
    load_reference,
)
from hypoxiapipe.api.schemas import (
    BatchScoreRequest,
    HealthResponse,
    ModelDetails,
    ModelSummary,
    RecurrenceInterpretation,
    RecurrencePrediction,
    RecurrencePredictionRequest,
    RecurrencePredictionResponse,
    RecurrenceProvenance,
    ReferenceScoreRequest,
    ReferenceSummary,
    ScoreResponse,
)
from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.modeling.bundle import BundleError, Smith20CoxBundle
from hypoxiapipe.scoring import methods as scoring_methods
from hypoxiapipe.scoring import score as score_matrix
from hypoxiapipe.signatures import registry

#: Below this, a cohort-relative score is not meaningfully estimated. Two
#: samples is enough to compute a standard deviation and nowhere near enough to
#: mean anything, so the floor is set where the estimate stops being absurd
#: rather than where it stops being arithmetically possible.
MIN_COHORT_SAMPLES = 20

#: Methods whose result depends on the other samples in the request. Every
#: method this package implements is in this set: `rowmean` and `weighted`
#: z-score each gene across samples, and `median_z` standardises the per-sample
#: median across samples. There is no sample-independent scoring method here,
#: which is precisely why /score/reference exists.
COHORT_RELATIVE_METHODS = set(scoring_methods.METHODS)
MAX_PREDICTION_BODY_BYTES = 2_000_000


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Verify configured model metadata and checksums before accepting traffic."""
    try:
        application.state.startup_model_ids = tuple(list_models())
    except ModelRegistryError:
        raise RuntimeError(
            "MODEL_INTEGRITY_ERROR: configured model failed startup verification"
        ) from None
    yield


app = FastAPI(
    title="hypoxiapipe scoring service",
    version=__version__,
    summary="Score transcriptomic hypoxia signatures, with provenance.",
    description=__doc__,
    lifespan=lifespan,
)


@app.middleware("http")
async def limit_prediction_body(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bound only trained-model requests; leave legacy scoring limits unchanged."""
    if request.url.path == "/predict/recurrence":
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isascii() or not declared.isdigit()):
            return JSONResponse(
                status_code=400,
                content={"detail": {"code": "INVALID_MATRIX", "message": "invalid content length"}},
            )
        if declared is not None and int(declared) > MAX_PREDICTION_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": "prediction body exceeds 2 MB",
                    }
                },
            )
        if len(await request.body()) > MAX_PREDICTION_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": "prediction body exceeds 2 MB",
                    }
                },
            )
    return await call_next(request)


_PREDICTOR_LIMITATION = (
    "The frozen STAR-native predictor has not undergone independent STAR-compatible "
    "external validation."
)


@app.exception_handler(RequestValidationError)
async def request_error(request: Request, exc: RequestValidationError) -> Response:
    """Keep existing validation responses while coding new prediction errors."""
    if request.url.path != "/predict/recurrence":
        return await request_validation_exception_handler(request, exc)
    locations = [tuple(error.get("loc", ())) for error in exc.errors()]
    if any("expression_type" in loc for loc in locations):
        code = "INCOMPATIBLE_EXPRESSION_TYPE"
    elif any("samples" in loc for loc in locations):
        code = "DUPLICATE_SAMPLE_IDS" if "duplicate" in str(exc).lower() else "INVALID_MATRIX"
    elif any("genes" in loc for loc in locations):
        code = "DUPLICATE_REQUIRED_GENES" if "duplicate" in str(exc).lower() else "INVALID_MATRIX"
    elif any("values" in loc for loc in locations) and "finite" in str(exc).lower():
        code = "NONFINITE_EXPRESSION"
    else:
        code = "INVALID_MATRIX"
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": code, "message": "invalid recurrence prediction request"}},
    )


def _registered_models() -> dict[str, Smith20CoxBundle]:
    try:
        return list_models()
    except ModelRegistryError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_INTEGRITY_ERROR", "message": str(exc)},
        ) from exc


def _model_summary(bundle: Smith20CoxBundle) -> ModelSummary:
    return ModelSummary(
        model_id=bundle.model_id,
        version=bundle.model_version,
        estimator=bundle.model["estimator"],
        endpoint=bundle.model["endpoint"],
        research_use_only=bundle.model["status"] == "research_use_only",
        expected_expression_type=bundle.features["expression_type"],
        n_features=len(bundle.features["genes"]),
        training_n=bundle.model["training_n"],
        training_events=bundle.model["training_events"],
        external_predictor_validation=bundle.metrics["frozen_predictor_external_validation"],
    )


def _model_or_404(model_id: str) -> Smith20CoxBundle:
    registered = _registered_models()
    if model_id not in registered:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": "model is not registered"},
        )
    return registered[model_id]


def _prediction_error(exc: BundleError) -> HTTPException:
    message = str(exc)
    if "incompatible or undeclared expression type" in message:
        code = "INCOMPATIBLE_EXPRESSION_TYPE"
    elif "missing Smith20 genes" in message:
        code = "MISSING_REQUIRED_GENES"
    elif "duplicated genes" in message:
        code = "DUPLICATE_REQUIRED_GENES"
    elif "finite" in message or "numeric" in message:
        code = "NONFINITE_EXPRESSION"
    elif "sample identifiers" in message:
        code = "DUPLICATE_SAMPLE_IDS"
    else:
        code = "INVALID_MATRIX"
    return HTTPException(status_code=422, detail={"code": code, "message": message})


def _signature_or_404(name: str) -> Any:
    try:
        return registry.load_bundled(name)
    except HypoxiapipeError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": str(exc),
                "available": sorted(registry.list_bundled()),
            },
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and what the service has loaded."""
    from hypoxiapipe.modeling import r_available  # noqa: PLC0415

    try:
        n_models, model_registry_error = len(list_models()), None
    except ModelRegistryError:
        n_models, model_registry_error = 0, "MODEL_INTEGRITY_ERROR"
    return HealthResponse(
        status="ok",
        version=__version__,
        n_signatures=len(registry.list_bundled()),
        n_references=len(list_references()),
        r_available=r_available(),
        n_models=n_models,
        models_ready=model_registry_error is None and n_models > 0,
        model_registry_error=model_registry_error,
        git_commit=os.environ.get("HYPOXIAPIPE_BUILD_GIT_SHA"),
        image_tag=os.environ.get("HYPOXIAPIPE_IMAGE_TAG"),
        base_image=os.environ.get("HYPOXIAPIPE_BASE_IMAGE"),
    )


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Report readiness without running a model prediction."""
    state = health()
    if state.model_registry_error:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_INTEGRITY_ERROR", "message": "configured model is not ready"},
        )
    return state


@app.get("/models", response_model=list[ModelSummary])
def models() -> list[ModelSummary]:
    """List only explicitly configured and checksum-verified research models."""
    return [_model_summary(bundle) for bundle in _registered_models().values()]


@app.get("/models/{model_id}", response_model=ModelDetails)
def model_details(model_id: str) -> ModelDetails:
    """Inspect safe training and validation metadata for one registered model."""
    bundle = _model_or_404(model_id)
    summary = _model_summary(bundle)
    return ModelDetails(
        **summary.model_dump(),
        input_contract=bundle.features["input_contract"],
        feature_genes=list(bundle.features["genes"]),
        internal_cv=bundle.metrics["internal_repeated_nested_cv"],
        censoring_horizon_months=bundle.model["censoring_horizon_months"],
        training_provenance={
            "population_hash": bundle.training["population_hash"],
            "expression_checksum": bundle.training["expression_checksum"],
            "signature_checksum": bundle.features["signature_checksum"],
            "scaler_checksum": bundle.scaler["checksum"],
            "clinical_table_checksum": bundle.provenance["clinical_table_checksum"],
            "cohort_manifest_checksum": bundle.provenance["cohort_manifest_checksum"],
        },
        external_signature_evidence=bundle.metrics["external_signature_evidence"],
        limitation=_PREDICTOR_LIMITATION,
    )


@app.post("/predict/recurrence", response_model=RecurrencePredictionResponse)
def predict_recurrence(request: RecurrencePredictionRequest) -> RecurrencePredictionResponse:
    """Apply a registered frozen model; research use only, no absolute risk."""
    bundle = _model_or_404(request.model_id)
    matrix = request.matrix.to_frame()
    if set(matrix.columns).intersection(bundle.features["genes"]) and not set(
        bundle.features["genes"]
    ).issubset(matrix.index):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_MATRIX",
                "message": "expected genes as rows, samples as columns",
            },
        )
    try:
        predicted = bundle.predict(matrix, expression_type=request.expression_type)
    except BundleError as exc:
        raise _prediction_error(exc) from exc
    rows = predicted.predictions.reset_index().rename(columns={"sample": "sample_id"})
    return RecurrencePredictionResponse(
        model_id=predicted.model_id,
        model_version=predicted.model_version,
        endpoint=bundle.model["endpoint"],
        research_use_only=True,
        expression_type=request.expression_type,
        predictions=[
            RecurrencePrediction.model_validate(row) for row in rows.to_dict(orient="records")
        ],
        provenance=RecurrenceProvenance(
            signature_checksum=bundle.features["signature_checksum"],
            scaler_checksum=predicted.provenance["scaler_checksum"],
            population_hash=predicted.provenance["population_hash"],
            manifest_checksum=predicted.provenance["manifest_checksum"],
            genes_used=list(predicted.genes_used),
        ),
        interpretation=RecurrenceInterpretation(
            relative_hazard=(
                "Relative to the Cox model reference score of zero; not an absolute "
                "recurrence probability."
            ),
            training_percentile=(
                "Position in the TCGA development score distribution; not a recurrence probability."
            ),
        ),
        limitation=_PREDICTOR_LIMITATION,
    )


@app.get("/signatures")
def signatures() -> list[dict[str, Any]]:
    """List the bundled signatures and their verified checksums."""
    out: list[dict[str, Any]] = []
    for name in sorted(registry.list_bundled()):
        try:
            sig = registry.load_bundled(name)
        except HypoxiapipeError as exc:
            out.append({"name": name, "error": str(exc)})
            continue
        out.append(
            {
                "name": sig.name,
                "n_genes": sig.n_genes,
                "scoring": sig.scoring,
                "checksum": sig.checksum,
                "source": sig.source,
                "weighted": sig.weighted,
            }
        )
    return out


@app.get("/references", response_model=list[ReferenceSummary])
def references() -> list[ReferenceSummary]:
    """List registered scoring references."""
    return [ReferenceSummary(**ref.describe()) for ref in list_references()]


@app.post("/score/batch", response_model=ScoreResponse)
def score_batch(request: BatchScoreRequest) -> ScoreResponse:
    """Score a complete cohort, relative to itself.

    Rejects submissions too small for a cohort-relative method to mean
    anything. Use ``/score/reference`` for one sample, or for a handful.
    """
    signature = _signature_or_404(request.signature)
    method = request.method or signature.scoring

    if method in COHORT_RELATIVE_METHODS and request.matrix.n_samples < MIN_COHORT_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": (
                    f"'{method}' is cohort-relative: each gene is standardised against the "
                    f"other samples in the request, so {request.matrix.n_samples} sample(s) "
                    "cannot produce a meaningful score. The same patient would score "
                    "differently alongside a different batch."
                ),
                "n_samples": request.matrix.n_samples,
                "minimum": MIN_COHORT_SAMPLES,
                "remedy": (
                    "Submit the full cohort, or use POST /score/reference to score "
                    "against frozen statistics from a named cohort."
                ),
            },
        )

    frame = request.matrix.to_frame()
    try:
        result = score_matrix(frame, signature, method=method)
    except HypoxiapipeError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc

    return ScoreResponse(
        scores={str(k): float(v) for k, v in result.scores.items()},
        signature=signature.name,
        signature_checksum=signature.checksum,
        method=result.method,
        n_samples=request.matrix.n_samples,
        n_genes_used=result.n_found,
        n_genes_expected=result.n_total,
        missing_genes=sorted(result.missing),
        relative_to="submitted cohort",
        provenance={
            "note": (
                "Scores are relative to the samples in this request; they are not "
                "comparable with scores computed over a different set of samples."
            )
        },
    )


@app.post("/score/reference", response_model=ScoreResponse)
def score_reference(request: ReferenceScoreRequest) -> ScoreResponse:
    """Score samples against frozen statistics from a registered reference.

    Valid for a single sample: the distribution comes from the reference
    cohort, not from the submission.
    """
    try:
        reference = load_reference(request.reference_id)
    except ReferenceError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc

    frame = request.matrix.to_frame()
    missing = [g for g in reference.genes if g not in frame.index]
    try:
        scores = reference.score(frame, strict=request.strict)
    except HypoxiapipeError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(exc),
                "missing_genes": missing[:20],
                "remedy": (
                    "Harmonise symbols to the reference's authority, or set strict=false "
                    "to score on the intersection and accept the reported coverage loss."
                ),
            },
        ) from exc

    return ScoreResponse(
        scores={str(k): float(v) for k, v in scores.items()},
        signature=reference.signature,
        signature_checksum=reference.signature_checksum,
        method=reference.method,
        n_samples=request.matrix.n_samples,
        n_genes_used=len(reference.genes) - len(missing),
        n_genes_expected=len(reference.genes),
        missing_genes=sorted(missing),
        relative_to=reference.reference_id,
        provenance=reference.describe(),
    )
