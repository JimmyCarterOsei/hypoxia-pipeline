"""Phase 8 tests: the scoring service.

The assertion that matters most is the negative one: a single-sample request to
``/score/batch`` must be refused. Every scoring method in this package is
cohort-relative, so a one-sample score is undefined - and an endpoint that
returned a plausible number for it would be the most dangerous thing here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="the API extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402

from hypoxiapipe.api.references import (  # noqa: E402
    REFERENCE_DIR_ENV,
    ReferenceError,
    build_reference,
    list_references,
    load_reference,
    save_reference,
)
from hypoxiapipe.api.schemas import ExpressionMatrix  # noqa: E402
from hypoxiapipe.signatures.registry import load_bundled  # noqa: E402

SIGNATURE = "smith20"


def cohort_frame(n_samples: int = 60, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic log-scale matrix covering the signature's genes."""
    genes = list(load_bundled(SIGNATURE).genes)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(8.0, 1.0, (len(genes), n_samples)),
        index=genes,
        columns=[f"S{i}" for i in range(n_samples)],
    )


def as_payload(frame: pd.DataFrame) -> dict:
    """Render a frame as an ExpressionMatrix request body."""
    return {
        "genes": [str(g) for g in frame.index],
        "samples": [str(s) for s in frame.columns],
        "values": frame.to_numpy().tolist(),
    }


@pytest.fixture
def references_dir(tmp_path, monkeypatch):
    """Point the service at a temporary reference directory holding one entry."""
    monkeypatch.setenv(REFERENCE_DIR_ENV, str(tmp_path))
    reference = build_reference(
        "demo-ref",
        cohort_frame(80, seed=1),
        load_bundled(SIGNATURE),
        cohort_name="DemoCohort",
        population_hash="sha256:demo",
    )
    save_reference(reference, tmp_path)
    return tmp_path


@pytest.fixture
def client(references_dir):
    from hypoxiapipe.api.app import app  # noqa: PLC0415

    return TestClient(app)


# --------------------------------------------------------------------------
# request validation
# --------------------------------------------------------------------------


def test_a_transposed_matrix_is_rejected_on_arrival():
    # Sending samples x genes would otherwise score the wrong axis silently.
    with pytest.raises(ValueError, match="genes x samples"):
        ExpressionMatrix(genes=["A", "B"], samples=["S1"], values=[[1.0]])


def test_ragged_rows_are_rejected():
    with pytest.raises(ValueError, match="values but 2 samples"):
        ExpressionMatrix(genes=["A"], samples=["S1", "S2"], values=[[1.0]])


def test_duplicate_identifiers_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ExpressionMatrix(genes=["A", "A"], samples=["S1"], values=[[1.0], [2.0]])


# --------------------------------------------------------------------------
# the constraint
# --------------------------------------------------------------------------


def test_a_single_sample_batch_request_is_refused(client):
    """The load-bearing behaviour of this service."""
    response = client.post(
        "/score/batch",
        json={"signature": SIGNATURE, "matrix": as_payload(cohort_frame(1))},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "cohort-relative" in detail["error"]
    # The error must point at the endpoint that can answer the question.
    assert "/score/reference" in detail["remedy"]


def test_a_small_batch_is_refused_with_the_threshold_named(client):
    response = client.post(
        "/score/batch",
        json={"signature": SIGNATURE, "matrix": as_payload(cohort_frame(5))},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["n_samples"] == 5
    assert response.json()["detail"]["minimum"] >= 2


def test_every_implemented_method_is_treated_as_cohort_relative():
    from hypoxiapipe.api.app import COHORT_RELATIVE_METHODS  # noqa: PLC0415
    from hypoxiapipe.scoring.methods import METHODS  # noqa: PLC0415

    # There is no sample-independent method here; if one is added, this test
    # forces a deliberate decision about whether it belongs in the set.
    assert set(METHODS) == COHORT_RELATIVE_METHODS


# --------------------------------------------------------------------------
# batch scoring
# --------------------------------------------------------------------------


def test_batch_scoring_returns_scores_and_provenance(client):
    frame = cohort_frame(60)
    response = client.post(
        "/score/batch", json={"signature": SIGNATURE, "matrix": as_payload(frame)}
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body["scores"]) == 60
    assert body["signature"] == SIGNATURE
    assert body["signature_checksum"].startswith("sha256:")
    assert body["n_genes_used"] == body["n_genes_expected"]
    assert body["relative_to"] == "submitted cohort"
    # The response says out loud that these scores are not portable.
    assert "not comparable" in body["provenance"]["note"]


def test_batch_scores_match_the_library(client):
    from hypoxiapipe.scoring import score  # noqa: PLC0415

    frame = cohort_frame(60)
    expected = score(frame, load_bundled(SIGNATURE))
    body = client.post(
        "/score/batch", json={"signature": SIGNATURE, "matrix": as_payload(frame)}
    ).json()
    for sample, value in expected.scores.items():
        assert body["scores"][str(sample)] == pytest.approx(float(value))


def test_an_unknown_signature_lists_the_available_ones(client):
    response = client.post(
        "/score/batch", json={"signature": "not-a-signature", "matrix": as_payload(cohort_frame())}
    )
    assert response.status_code == 404
    assert SIGNATURE in response.json()["detail"]["available"]


# --------------------------------------------------------------------------
# reference scoring
# --------------------------------------------------------------------------


def test_one_sample_scores_against_a_reference(client):
    """The whole reason references exist."""
    one = cohort_frame(1, seed=9)
    response = client.post(
        "/score/reference", json={"reference_id": "demo-ref", "matrix": as_payload(one)}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["scores"]) == 1
    assert body["relative_to"] == "demo-ref"
    assert body["provenance"]["population_hash"] == "sha256:demo"
    assert body["provenance"]["cohort"] == "DemoCohort"


def test_a_reference_score_does_not_depend_on_the_other_samples_submitted(client):
    """The property that makes single-sample scoring valid at all."""
    frame = cohort_frame(10, seed=4)
    alone = client.post(
        "/score/reference",
        json={"reference_id": "demo-ref", "matrix": as_payload(frame.iloc[:, [0]])},
    ).json()
    together = client.post(
        "/score/reference", json={"reference_id": "demo-ref", "matrix": as_payload(frame)}
    ).json()
    assert alone["scores"]["S0"] == pytest.approx(together["scores"]["S0"], rel=1e-9)


def test_batch_scores_do_depend_on_the_other_samples(client):
    """The contrast that justifies the two endpoints."""
    from hypoxiapipe.scoring import score  # noqa: PLC0415

    frame = cohort_frame(60, seed=5)
    full = score(frame, load_bundled(SIGNATURE)).scores["S0"]
    half = score(frame.iloc[:, :30], load_bundled(SIGNATURE)).scores["S0"]
    assert full != pytest.approx(half, rel=1e-6), (
        "if this ever passes, scoring stopped being cohort-relative and the "
        "batch endpoint's restriction should be revisited"
    )


def test_a_missing_gene_is_refused_unless_strict_is_relaxed(client):
    frame = cohort_frame(5, seed=6).iloc[1:]  # drop one signature gene
    strict = client.post(
        "/score/reference", json={"reference_id": "demo-ref", "matrix": as_payload(frame)}
    )
    assert strict.status_code == 422
    assert strict.json()["detail"]["missing_genes"]
    assert "strict=false" in strict.json()["detail"]["remedy"]

    relaxed = client.post(
        "/score/reference",
        json={"reference_id": "demo-ref", "matrix": as_payload(frame), "strict": False},
    )
    assert relaxed.status_code == 200
    assert relaxed.json()["n_genes_used"] == relaxed.json()["n_genes_expected"] - 1
    assert relaxed.json()["missing_genes"]


def test_an_unknown_reference_lists_what_is_registered(client):
    response = client.post(
        "/score/reference", json={"reference_id": "absent", "matrix": as_payload(cohort_frame(1))}
    )
    assert response.status_code == 404
    assert "demo-ref" in response.json()["detail"]["error"]


# --------------------------------------------------------------------------
# the reference store
# --------------------------------------------------------------------------


def test_a_reference_round_trips_and_detects_tampering(tmp_path):
    import json  # noqa: PLC0415

    reference = build_reference(
        "rt", cohort_frame(40), load_bundled(SIGNATURE), cohort_name="C", population_hash="h"
    )
    path = save_reference(reference, tmp_path)
    assert load_reference("rt", tmp_path).scaler.checksum == reference.scaler.checksum

    raw = json.loads(path.read_text())
    first = raw["scaler"]["genes"][0]
    raw["scaler"]["means"][first] += 1.0
    path.write_text(json.dumps(raw))
    with pytest.raises(Exception, match="checksum mismatch"):
        load_reference("rt", tmp_path)


def test_listing_an_absent_directory_is_empty_not_an_error(tmp_path):
    assert list_references(tmp_path / "nope") == []


def test_a_reference_needs_at_least_one_signature_gene(tmp_path):
    unrelated = pd.DataFrame(
        np.zeros((3, 10)), index=["X1", "X2", "X3"], columns=[f"S{i}" for i in range(10)]
    )
    with pytest.raises(ReferenceError, match="none of"):
        build_reference("bad", unrelated, load_bundled(SIGNATURE), cohort_name="C")


# --------------------------------------------------------------------------
# service metadata
# --------------------------------------------------------------------------


def test_health_reports_what_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_signatures"] >= 1
    assert body["n_references"] == 1


def test_signatures_endpoint_exposes_checksums(client):
    body = client.get("/signatures").json()
    entry = next(s for s in body if s.get("name") == SIGNATURE)
    assert entry["checksum"].startswith("sha256:")
    assert entry["n_genes"] == load_bundled(SIGNATURE).n_genes


def test_references_endpoint_lists_provenance(client):
    body = client.get("/references").json()
    assert len(body) == 1
    assert body[0]["reference_id"] == "demo-ref"
    assert body[0]["population_hash"] == "sha256:demo"
