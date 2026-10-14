"""Phase 9E frozen research-model bundle and inference contract checks."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.modeling.bundle import (
    EXPRESSION_TYPE,
    BundleError,
    load_model_bundle,
)
from hypoxiapipe.provenance.hashing import hash_file

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "prad-bcr-star-smith20-cox-v1"
KNOWN = ROOT / "tests" / "fixtures" / "prad_bcr_star_v1_known_sample.json"


@pytest.fixture
def known_expression():
    raw = json.loads(KNOWN.read_text())
    values = pd.DataFrame({raw["patient"]: raw["gene_values"]})
    values.attrs["expression_type"] = EXPRESSION_TYPE
    return values, raw


@pytest.fixture
def copied_bundle(tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(BUNDLE, target)
    return target


def _rehash_file(bundle, name):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"][name] = hash_file(bundle / name)
    (bundle / "manifest.json").write_text(json.dumps(manifest))


def test_model_bundle_roundtrip_and_archived_training_sample(copied_bundle, known_expression):
    expression, known = known_expression
    before = load_model_bundle(BUNDLE).predict(expression)
    after = load_model_bundle(copied_bundle).predict(expression)
    pd.testing.assert_frame_equal(before.predictions, after.predictions)
    assert before.model_id == "prad-bcr-star-smith20-cox-v1"
    assert before.model_version == "1.0.0"
    assert before.predictions.iloc[0].linear_predictor == pytest.approx(
        known["archived_linear_predictor"], abs=1e-10
    )
    assert after.predictions.iloc[0].relative_hazard == pytest.approx(
        math.exp(known["archived_linear_predictor"]), abs=1e-10
    )


def test_manifest_detects_tampering(copied_bundle):
    scaler_path = copied_bundle / "scaler.json"
    data = json.loads(scaler_path.read_text())
    gene = data["genes"][0]
    data["gene_means"][gene] += 1
    scaler_path.write_text(json.dumps(data))
    with pytest.raises(BundleError, match="manifest checksum mismatch"):
        load_model_bundle(copied_bundle)


def test_cross_file_scaler_checksum_detects_rehashed_tampering(copied_bundle):
    scaler_path = copied_bundle / "scaler.json"
    data = json.loads(scaler_path.read_text())
    gene = data["genes"][0]
    data["gene_means"][gene] += 1
    scaler_path.write_text(json.dumps(data))
    _rehash_file(copied_bundle, "scaler.json")
    with pytest.raises(BundleError, match="model/scaler checksum mismatch"):
        load_model_bundle(copied_bundle)


def test_missing_and_duplicate_required_genes_fail(known_expression):
    expression, _ = known_expression
    model = load_model_bundle(BUNDLE)
    with pytest.raises(BundleError, match="missing Smith20 genes"):
        model.predict(expression.iloc[1:])
    duplicate = pd.concat([expression, expression.iloc[[0]]])
    with pytest.raises(BundleError, match="duplicated genes"):
        model.predict(duplicate)


def test_extra_genes_sample_order_and_feature_row_order_do_not_change_predictions(
    known_expression,
):
    expression, _ = known_expression
    model = load_model_bundle(BUNDLE)
    second = expression.copy()
    second.columns = ["another-sample"]
    second.iloc[:, 0] += 0.1
    values = pd.concat([expression, second], axis=1)
    expected = model.predict(values).predictions
    extra = pd.concat(
        [values.iloc[::-1], pd.DataFrame([[999, 999]], index=["EXTRA"], columns=values.columns)]
    )
    actual = model.predict(extra.loc[:, values.columns[::-1]], expression_type=EXPRESSION_TYPE)
    pd.testing.assert_frame_equal(expected.sort_index(), actual.predictions.sort_index())


def test_inference_is_frozen_and_batch_independent(known_expression):
    expression, _ = known_expression
    model = load_model_bundle(BUNDLE)
    first = model.predict(expression).predictions.iloc[0]
    shifted = expression.copy()
    shifted.columns = ["unrelated"]
    shifted.iloc[:, 0] += 2
    batch = pd.concat([expression, shifted], axis=1)
    again = model.predict(batch).predictions.loc[expression.columns[0]]
    pd.testing.assert_series_equal(first, again, check_names=False)
    assert not hasattr(model, "fit")
    assert not hasattr(model, "refit")
    assert model.scaler["checksum"] == load_model_bundle(BUNDLE).scaler["checksum"]


def test_incompatible_or_undeclared_expression_type_fails(known_expression):
    expression, _ = known_expression
    model = load_model_bundle(BUNDLE)
    with pytest.raises(BundleError, match="incompatible or undeclared"):
        model.predict(expression, expression_type="deseq2_vst")
    expression.attrs.clear()
    with pytest.raises(BundleError, match="incompatible or undeclared"):
        model.predict(expression)


def test_nonfinite_values_and_duplicate_samples_fail(known_expression):
    expression, _ = known_expression
    model = load_model_bundle(BUNDLE)
    broken = expression.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(BundleError, match="finite"):
        model.predict(broken)
    duplicated = pd.concat([expression, expression], axis=1)
    with pytest.raises(BundleError, match="unique strings"):
        model.predict(duplicated)


def test_feature_order_and_signature_checksum_enforced(copied_bundle):
    path = copied_bundle / "features.json"
    data = json.loads(path.read_text())
    data["genes"][0], data["genes"][1] = data["genes"][1], data["genes"][0]
    path.write_text(json.dumps(data))
    _rehash_file(copied_bundle, "features.json")
    with pytest.raises(BundleError, match="feature ordering"):
        load_model_bundle(copied_bundle)


def test_metadata_keeps_signature_evidence_separate_from_predictor_validation():
    model = load_model_bundle(BUNDLE)
    metrics = model.metrics
    assert metrics["frozen_predictor_external_validation"] is False
    assert (
        "not external validation of the frozen STAR-native predictor"
        in metrics["external_signature_evidence"]["interpretation"]
    )
    assert set(metrics["external_signature_evidence"]["cohorts"]) == {
        "MSKCC",
        "Cambridge",
        "Stockholm",
    }
    assert model.model["absolute_recurrence_probability_supported"] is False
    assert "recurrence_probability" not in model.model["outputs"]
