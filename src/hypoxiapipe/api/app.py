"""The scoring service.

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
indistinguishable from a real result.

Run it with::

    HYPOXIAPIPE_REFERENCES=references uvicorn hypoxiapipe.api.app:app
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from hypoxiapipe import __version__
from hypoxiapipe.api.references import (
    ReferenceError,
    list_references,
    load_reference,
)
from hypoxiapipe.api.schemas import (
    BatchScoreRequest,
    HealthResponse,
    ReferenceScoreRequest,
    ReferenceSummary,
    ScoreResponse,
)
from hypoxiapipe.errors import HypoxiapipeError
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

app = FastAPI(
    title="hypoxiapipe scoring service",
    version=__version__,
    summary="Score transcriptomic hypoxia signatures, with provenance.",
    description=__doc__,
)


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

    return HealthResponse(
        status="ok",
        version=__version__,
        n_signatures=len(registry.list_bundled()),
        n_references=len(list_references()),
        r_available=r_available(),
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
