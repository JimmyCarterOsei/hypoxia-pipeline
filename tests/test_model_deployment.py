"""Deployment startup, readiness and packaged-model safeguards."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi", reason="the API extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402

from hypoxiapipe import __version__  # noqa: E402
from hypoxiapipe.api.app import app  # noqa: E402
from hypoxiapipe.api.models import MODELS_DIR_ENV  # noqa: E402
from hypoxiapipe.modeling.bundle import Smith20CoxBundle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "prad-bcr-star-smith20-cox-v1"


def test_startup_verifies_registered_model_without_predicting(monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, str(ROOT / "models"))

    def forbidden_predict(*_args, **_kwargs):
        raise AssertionError("startup must not run model inference")

    monkeypatch.setattr(Smith20CoxBundle, "predict", forbidden_predict)
    with TestClient(app) as client:
        assert app.state.startup_model_ids == (MODEL_ID,)
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["service"] == "hypoxiapipe"
        assert health["models_ready"] is True
        assert health["n_models"] == 1
        assert client.get("/ready").status_code == 200


def test_absent_or_empty_model_registry_keeps_scoring_ready(monkeypatch, tmp_path):
    for location in (None, tmp_path / "empty"):
        if location is None:
            monkeypatch.delenv(MODELS_DIR_ENV, raising=False)
        else:
            location.mkdir()
            monkeypatch.setenv(MODELS_DIR_ENV, str(location))
        with TestClient(app) as client:
            state = client.get("/ready").json()
            assert state["status"] == "ok"
            assert state["models_ready"] is False
            assert state["n_models"] == 0
            assert client.get("/models").json() == []
            assert client.get("/signatures").status_code == 200


def test_corrupt_bundle_blocks_startup_and_readiness_without_path_leak(monkeypatch, tmp_path):
    root = tmp_path / "models"
    shutil.copytree(ROOT / "models" / MODEL_ID, root / MODEL_ID)
    payload = root / MODEL_ID / "model.json"
    payload.write_text(payload.read_text().replace("0.5806182816932", "0.7806182816932"))
    monkeypatch.setenv(MODELS_DIR_ENV, str(root))
    with pytest.raises(RuntimeError, match="MODEL_INTEGRITY_ERROR") as error, TestClient(app):
        pass
    assert str(tmp_path) not in str(error.value)
    # Without lifespan startup, the endpoint still gives a controlled 503.
    client = TestClient(app)
    readiness = client.get("/ready")
    assert readiness.status_code == 503
    assert readiness.json()["detail"]["code"] == "MODEL_INTEGRITY_ERROR"


def test_software_and_build_metadata_are_separate_from_model_version(monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, str(ROOT / "models"))
    monkeypatch.setenv("HYPOXIAPIPE_BUILD_GIT_SHA", "abc123")
    monkeypatch.setenv("HYPOXIAPIPE_IMAGE_TAG", "sha-abc123")
    monkeypatch.setenv("HYPOXIAPIPE_BASE_IMAGE", "python:3.12-slim-bookworm@sha256:demo")
    with TestClient(app) as client:
        health = client.get("/health").json()
        model = client.get("/models").json()[0]
    assert health["version"] == __version__
    assert health["git_commit"] == "abc123"
    assert health["image_tag"] == "sha-abc123"
    assert health["base_image"].endswith("sha256:demo")
    assert model["version"] == "1.0.0"


def test_prediction_body_limit_and_invalid_json_return_controlled_errors(monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, str(ROOT / "models"))
    client = TestClient(app)
    oversized = client.post(
        "/predict/recurrence",
        content=b"x" * 2_000_001,
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert oversized.json()["detail"]["code"] == "REQUEST_TOO_LARGE"
    malformed = client.post(
        "/predict/recurrence", content=b"{", headers={"Content-Type": "application/json"}
    )
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["code"] == "INVALID_MATRIX"


def test_api_image_contains_only_inference_runtime_and_frozen_bundle():
    dockerfile = (ROOT / "docker/Dockerfile.api").read_text()
    assert "HYPOXIAPIPE_MODELS=/app/models" in dockerfile
    assert "HYPOXIAPIPE_REFERENCES=/app/references" in dockerfile
    assert "COPY --chown=api:api models/prad-bcr-star-smith20-cox-v1" in dockerfile
    assert "COPY --chown=api:api references /app/references" in dockerfile
    assert "USER api" in dockerfile
    assert 'CMD ["uvicorn", "hypoxiapipe.api.app:app"' in dockerfile
    assert "pip install --no-deps ." in dockerfile
    assert "'fastapi==" in dockerfile and "'uvicorn==" in dockerfile
    assert "'scikit-survival==" not in dockerfile


def test_ci_image_workflow_and_example_are_parseable():
    workflow = yaml.safe_load((ROOT / ".github/workflows/api-image.yml").read_text())
    assert "build-and-smoke" in workflow["jobs"]
    steps = json.dumps(workflow["jobs"]["build-and-smoke"]["steps"])
    assert "demo_model_api.py --known-fixture" in steps
    assert "docker buildx build" in steps
    assert "BASE_DIGEST" in steps
    example = json.loads((ROOT / "docs/examples/recurrence-prediction.json").read_text())
    assert len(example["matrix"]["genes"]) == 20
