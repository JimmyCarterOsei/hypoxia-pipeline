"""Phase 9G API regression and frozen inference identity checks."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="the API extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from hypoxiapipe.api.app import app  # noqa: E402
from hypoxiapipe.api.models import MODELS_DIR_ENV  # noqa: E402
from hypoxiapipe.api.references import (  # noqa: E402
    REFERENCE_DIR_ENV,
    build_reference,
    save_reference,
)
from hypoxiapipe.cli import app as cli_app  # noqa: E402
from hypoxiapipe.modeling.bundle import EXPRESSION_TYPE, load_model_bundle  # noqa: E402
from hypoxiapipe.signatures.registry import load_bundled  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "prad-bcr-star-smith20-cox-v1"
BUNDLE = ROOT / "models" / MODEL_ID
KNOWN = ROOT / "tests" / "fixtures" / "prad_bcr_star_v1_known_sample.json"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv(MODELS_DIR_ENV, str(ROOT / "models"))
    monkeypatch.setenv(REFERENCE_DIR_ENV, str(tmp_path / "empty-references"))
    return TestClient(app)


@pytest.fixture
def frame():
    known = json.loads(KNOWN.read_text())
    return pd.DataFrame({known["patient"]: known["gene_values"]})


def _payload(frame: pd.DataFrame, *, model_id: str = MODEL_ID, domain: str = EXPRESSION_TYPE):
    return {
        "model_id": model_id,
        "expression_type": domain,
        "matrix": {
            "genes": list(frame.index),
            "samples": list(frame.columns),
            "values": frame.to_numpy().tolist(),
        },
    }


def _post(client: TestClient, frame: pd.DataFrame):
    return client.post("/predict/recurrence", json=_payload(frame))


def _prediction(client: TestClient, frame: pd.DataFrame):
    response = _post(client, frame)
    assert response.status_code == 200, response.text
    return response.json()["predictions"]


def test_models_listing_and_health(client):
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 1
    assert models[0] == {
        "model_id": MODEL_ID,
        "version": "1.0.0",
        "estimator": "fixed_smith20_cox",
        "endpoint": "longitudinal TCGA-PRAD biochemical recurrence",
        "research_use_only": True,
        "expected_expression_type": EXPRESSION_TYPE,
        "n_features": 20,
        "training_n": 483,
        "training_events": 97,
        "external_predictor_validation": False,
    }
    assert client.get("/health").json()["n_models"] == 1


def test_model_details_are_safe_and_distinguish_external_evidence(client):
    response = client.get(f"/models/{MODEL_ID}")
    assert response.status_code == 200
    detail = response.json()
    assert len(detail["feature_genes"]) == 20
    assert detail["input_contract"]["expression_type"] == EXPRESSION_TYPE
    assert detail["internal_cv"]["mean"] == pytest.approx(0.6825397634337843)
    assert detail["training_provenance"]["population_hash"].startswith("sha256:")
    assert (
        "not external validation of the frozen STAR-native predictor"
        in detail["external_signature_evidence"]["interpretation"]
    )
    assert "not undergone independent" in detail["limitation"]
    assert "/Users/" not in response.text


def test_unknown_model_and_path_like_id_are_not_loaded(client, frame):
    for model_id in ("absent", "../../another-bundle", "https://example.org/model"):
        response = client.post("/predict/recurrence", json=_payload(frame, model_id=model_id))
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "MODEL_NOT_FOUND"
        assert "/Users/" not in response.text
    assert client.get("/models/absent").status_code == 404


def test_known_single_sample_prediction_and_research_limits(client, frame):
    response = _post(client, frame)
    assert response.status_code == 200
    body = response.json()
    row = body["predictions"][0]
    assert row["sample_id"] == "TCGA-2A-A8VL"
    assert row["smith20_score"] == pytest.approx(-0.9994416998626602, abs=1e-12)
    assert row["linear_predictor"] == pytest.approx(-0.5802941224267887, abs=1e-12)
    assert row["relative_hazard"] == pytest.approx(0.5597337121145431, abs=1e-12)
    assert row["training_percentile"] == pytest.approx(15.320910973084887, abs=1e-12)
    assert body["research_use_only"] is True
    assert body["expression_type"] == EXPRESSION_TYPE
    assert len(body["provenance"]["genes_used"]) == 20
    assert "not an absolute" in body["interpretation"]["relative_hazard"]
    assert "independent STAR-compatible" in body["limitation"]
    assert "recurrence_probability" not in response.text


def test_api_cli_and_direct_bundle_predictions_are_numerically_identical(client, frame, tmp_path):
    api_row = _prediction(client, frame)[0]
    direct = load_model_bundle(BUNDLE).predict(frame, expression_type=EXPRESSION_TYPE)
    tsv = tmp_path / "patient.tsv"
    frame.to_csv(tsv, sep="\t", index_label="gene")
    cli = CliRunner().invoke(
        cli_app,
        [
            "model",
            "predict",
            "--model",
            str(BUNDLE),
            "--expression",
            str(tsv),
            "--expression-type",
            EXPRESSION_TYPE,
            "--format",
            "json",
        ],
    )
    assert cli.exit_code == 0, cli.output
    cli_row = json.loads(cli.stdout)["predictions"][0]
    direct_row = direct.predictions.iloc[0]
    for name in ("smith20_score", "linear_predictor", "relative_hazard", "training_percentile"):
        assert api_row[name] == pytest.approx(cli_row[name], abs=1e-12)
        assert api_row[name] == pytest.approx(float(direct_row[name]), abs=1e-12)


def test_batch_and_both_axis_orders_preserve_each_sample(client, frame):
    others = []
    for index, offset in enumerate((0.1, -0.1, 0.25), start=1):
        shifted = frame.copy() + offset
        shifted.columns = [f"other-{index}"]
        others.append(shifted)
    batch = pd.concat([frame, *others], axis=1)
    alone = _prediction(client, frame)[0]
    together = _prediction(client, batch)
    reordered = _prediction(client, batch.iloc[::-1, ::-1])
    assert len(together) == 4
    assert together[0] == alone
    assert {row["sample_id"]: row for row in together} == {
        row["sample_id"]: row for row in reordered
    }


def test_extra_genes_are_ignored(client, frame):
    extra = pd.concat([frame, pd.DataFrame({frame.columns[0]: [999]}, index=["EXTRA"])])
    assert _prediction(client, extra) == _prediction(client, frame)


def test_missing_and_duplicate_required_genes_are_structured_errors(client, frame):
    missing = _post(client, frame.iloc[1:])
    assert missing.status_code == 422
    assert missing.json()["detail"]["code"] == "MISSING_REQUIRED_GENES"
    duplicated = _post(client, pd.concat([frame, frame.iloc[[0]]]))
    assert duplicated.status_code == 422
    assert duplicated.json()["detail"]["code"] == "DUPLICATE_REQUIRED_GENES"


def test_nonfinite_expression_is_rejected(client, frame):
    request = _payload(frame)
    request["matrix"]["values"][0][0] = "NaN"
    response = client.post("/predict/recurrence", json=request)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "NONFINITE_EXPRESSION"


def test_duplicate_samples_and_zero_samples_are_rejected(client, frame):
    request = _payload(pd.concat([frame, frame], axis=1))
    duplicate = client.post("/predict/recurrence", json=request)
    assert duplicate.status_code == 422
    assert duplicate.json()["detail"]["code"] == "DUPLICATE_SAMPLE_IDS"
    request["matrix"]["samples"] = []
    request["matrix"]["values"] = [[] for _ in request["matrix"]["genes"]]
    empty = client.post("/predict/recurrence", json=request)
    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "INVALID_MATRIX"


def test_wrong_or_undeclared_expression_type_is_rejected(client, frame):
    for domain in ("microarray", "deseq2_vst", "rsem", "fpkm", "raw_counts"):
        response = client.post("/predict/recurrence", json=_payload(frame, domain=domain))
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "INCOMPATIBLE_EXPRESSION_TYPE"
    request = _payload(frame)
    del request["expression_type"]
    response = client.post("/predict/recurrence", json=request)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INCOMPATIBLE_EXPRESSION_TYPE"


def test_malformed_or_transposed_matrix_is_rejected(client, frame):
    malformed = _payload(frame)
    malformed["matrix"]["values"].pop()
    response = client.post("/predict/recurrence", json=malformed)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_MATRIX"
    transposed = client.post("/predict/recurrence", json=_payload(frame.T))
    assert transposed.status_code == 422
    assert transposed.json()["detail"]["code"] == "INVALID_MATRIX"


def test_corrupt_configured_bundle_fails_loudly_without_breaking_scoring(
    client, frame, tmp_path, monkeypatch
):
    root = tmp_path / "models"
    shutil.copytree(BUNDLE, root / MODEL_ID)
    model_file = root / MODEL_ID / "model.json"
    model_file.write_text(model_file.read_text().replace("0.5806182816932", "0.7806182816932"))
    monkeypatch.setenv(MODELS_DIR_ENV, str(root))
    for response in (client.get("/models"), _post(client, frame)):
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "MODEL_INTEGRITY_ERROR"
        assert str(tmp_path) not in response.text
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_registry_error"] == "MODEL_INTEGRITY_ERROR"
    scoring = client.post(
        "/score/batch", json={"signature": "smith20", "matrix": _payload(frame)["matrix"]}
    )
    assert scoring.status_code == 422
    assert "cohort-relative" in scoring.json()["detail"]["error"]


def test_no_model_directory_keeps_scoring_service_healthy(client, monkeypatch):
    monkeypatch.delenv(MODELS_DIR_ENV, raising=False)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["n_models"] == 0
    assert client.get("/models").json() == []


def test_existing_reference_scoring_works_with_model_configured(
    client, frame, tmp_path, monkeypatch
):
    monkeypatch.setenv(REFERENCE_DIR_ENV, str(tmp_path))
    signature = load_bundled("smith20")
    training = pd.concat([frame + index * 0.1 for index in range(30)], axis=1)
    training.columns = [f"S{index}" for index in range(30)]
    reference = build_reference("phase9g-ref", training, signature, "synthetic")
    save_reference(reference, tmp_path)
    response = client.post(
        "/score/reference",
        json={"reference_id": "phase9g-ref", "matrix": _payload(frame)["matrix"]},
    )
    assert response.status_code == 200
    assert response.json()["relative_to"] == "phase9g-ref"


def test_openapi_has_request_response_schemas_and_valid_example(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    operation = spec["paths"]["/predict/recurrence"]["post"]
    assert "RecurrencePredictionRequest" in json.dumps(operation)
    assert "RecurrencePredictionResponse" in json.dumps(operation)
    schemas = spec["components"]["schemas"]
    request_schema = schemas["RecurrencePredictionRequest"]
    example = request_schema["examples"][0]
    assert len(example["matrix"]["genes"]) == 20
    assert len(example["matrix"]["values"]) == 20
    assert example["expression_type"] == EXPRESSION_TYPE
    assert len(schemas["RecurrencePredictionResponse"]["properties"]) >= 8
    assert client.post("/predict/recurrence", json=example).status_code == 200
    readme_example = json.loads((ROOT / "docs/examples/recurrence-prediction.json").read_text())
    assert client.post("/predict/recurrence", json=readme_example).status_code == 200
