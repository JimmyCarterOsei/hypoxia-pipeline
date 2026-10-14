"""End-to-end CLI checks for the frozen STAR Smith20 research predictor."""

import io
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hypoxiapipe.cli import app
from hypoxiapipe.modeling.bundle import EXPRESSION_TYPE

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "prad-bcr-star-smith20-cox-v1"
KNOWN = ROOT / "tests" / "fixtures" / "prad_bcr_star_v1_known_sample.json"
RUNNER = CliRunner()


@pytest.fixture
def matrix() -> pd.DataFrame:
    raw = json.loads(KNOWN.read_text())
    return pd.DataFrame({raw["patient"]: raw["gene_values"]})


def _write(matrix: pd.DataFrame, path: Path) -> Path:
    matrix.to_csv(path, sep="\t", index_label="gene")
    return path


def _predict(path: Path, *extra: str, model: Path = BUNDLE):
    return RUNNER.invoke(
        app,
        [
            "model",
            "predict",
            "--model",
            str(model),
            "--expression",
            str(path),
            "--expression-type",
            EXPRESSION_TYPE,
            *extra,
        ],
    )


def _json(path: Path, *extra: str, model: Path = BUNDLE) -> dict:
    result = _predict(path, "--format", "json", *extra, model=model)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_single_sample_cli_prediction_matches_archived_values(matrix, tmp_path):
    data = _json(_write(matrix, tmp_path / "single.tsv"))
    row = data["predictions"][0]
    assert row["sample_id"] == "TCGA-2A-A8VL"
    assert row["smith20_score"] == pytest.approx(-0.999441699862660)
    assert row["linear_predictor"] == pytest.approx(-0.580294122426789)
    assert row["relative_hazard"] == pytest.approx(0.559733712114543)
    assert row["training_percentile"] == pytest.approx(15.320910973084887)


def test_csv_input_and_default_human_table(matrix, tmp_path):
    path = tmp_path / "single.csv"
    matrix.to_csv(path, index_label="gene")
    result = _predict(path)
    assert result.exit_code == 0, result.output
    assert "smith20_score" in result.stdout
    assert "TCGA-2A-A8VL" in result.stdout
    assert "Research use only" in result.stderr


def test_multi_sample_batch_predictions_match_single_and_reordered(matrix, tmp_path):
    second = matrix.copy() + 0.2
    second.columns = ["second"]
    batch = pd.concat([matrix, second], axis=1)
    alone = _json(_write(matrix, tmp_path / "alone.tsv"))["predictions"][0]
    together = _json(_write(batch, tmp_path / "batch.tsv"))["predictions"]
    reversed_rows = _json(_write(batch.iloc[:, ::-1], tmp_path / "reversed.tsv"))["predictions"]
    assert len(together) == 2
    assert together[0] == alone
    assert {row["sample_id"]: row for row in together} == {
        row["sample_id"]: row for row in reversed_rows
    }


def test_gene_row_order_and_extra_genes_do_not_change_predictions(matrix, tmp_path):
    original = _json(_write(matrix, tmp_path / "original.tsv"))["predictions"]
    changed = pd.concat(
        [matrix.iloc[::-1], pd.DataFrame({matrix.columns[0]: [999]}, index=["EXTRA"])]
    )
    actual = _json(_write(changed, tmp_path / "changed.tsv"))["predictions"]
    assert actual == original


def test_missing_required_gene_fails(matrix, tmp_path):
    result = _predict(_write(matrix.iloc[1:], tmp_path / "missing.tsv"))
    assert result.exit_code != 0
    assert "missing Smith20 genes" in result.output


def test_duplicate_required_gene_fails(matrix, tmp_path):
    duplicated = pd.concat([matrix, matrix.iloc[[0]]])
    result = _predict(_write(duplicated, tmp_path / "duplicate.tsv"))
    assert result.exit_code != 0
    assert "duplicated genes" in result.output


def test_nonfinite_required_value_fails(matrix, tmp_path):
    broken = matrix.copy()
    broken.iloc[0, 0] = np.inf
    result = _predict(_write(broken, tmp_path / "infinite.tsv"))
    assert result.exit_code != 0
    assert "must be finite" in result.output


@pytest.mark.parametrize("domain", ["deseq2_vst", "microarray", "rsem", "fpkm", "raw_counts"])
def test_incompatible_expression_domain_fails(matrix, tmp_path, domain):
    path = _write(matrix, tmp_path / "input.tsv")
    result = RUNNER.invoke(
        app,
        [
            "model",
            "predict",
            "--model",
            str(BUNDLE),
            "--expression",
            str(path),
            "--expression-type",
            domain,
        ],
    )
    assert result.exit_code != 0
    assert "incompatible or undeclared expression type" in result.output


def test_undeclared_expression_domain_fails(matrix, tmp_path):
    path = _write(matrix, tmp_path / "input.tsv")
    result = RUNNER.invoke(
        app, ["model", "predict", "--model", str(BUNDLE), "--expression", str(path)]
    )
    assert result.exit_code != 0
    plain_output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    assert "--expression-type" in plain_output


def test_corrupted_bundle_fails(matrix, tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copied)
    model = copied / "model.json"
    model.write_text(model.read_text().replace("0.5806182816932", "0.7806182816932"))
    result = _predict(_write(matrix, tmp_path / "input.tsv"), model=copied)
    assert result.exit_code != 0
    assert "manifest checksum mismatch" in result.output


def test_missing_bundle_file_fails_with_useful_error(matrix, tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copied)
    (copied / "scaler.json").unlink()
    result = _predict(_write(matrix, tmp_path / "input.tsv"), model=copied)
    assert result.exit_code != 0
    assert "missing or unreadable bundle file: scaler.json" in result.output


def test_json_output_schema_and_provenance(matrix, tmp_path):
    data = _json(_write(matrix, tmp_path / "input.tsv"))
    assert set(data) == {"metadata", "predictions"}
    assert set(data["predictions"][0]) == {
        "sample_id",
        "smith20_score",
        "linear_predictor",
        "relative_hazard",
        "training_percentile",
        "model_id",
        "model_version",
    }
    assert data["metadata"]["expression_type"] == EXPRESSION_TYPE
    assert len(data["metadata"]["genes_used"]) == 20
    for name in (
        "signature_checksum",
        "scaler_checksum",
        "population_hash",
        "bundle_manifest_checksum",
        "timestamp_utc",
    ):
        assert data["metadata"][name]
    assert "recurrence_probability" not in json.dumps(data)


def test_tsv_output_schema(matrix, tmp_path):
    path = _write(matrix, tmp_path / "input.tsv")
    result = _predict(path, "--format", "tsv")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(io.StringIO(result.stdout), sep="\t")
    assert table.columns.tolist() == [
        "sample_id",
        "smith20_score",
        "linear_predictor",
        "relative_hazard",
        "training_percentile",
        "model_id",
        "model_version",
    ]


def test_output_file_and_provenance_sidecar(matrix, tmp_path):
    path = _write(matrix, tmp_path / "input.tsv")
    output = tmp_path / "predictions.tsv"
    metadata = tmp_path / "metadata.json"
    result = _predict(
        path, "--format", "tsv", "--output", str(output), "--metadata-output", str(metadata)
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert pd.read_csv(output, sep="\t").shape == (1, 7)
    assert json.loads(metadata.read_text())["model_id"] == "prad-bcr-star-smith20-cox-v1"


def test_output_cannot_overwrite_input_or_model(matrix, tmp_path):
    path = _write(matrix, tmp_path / "input.tsv")
    result = _predict(path, "--output", str(path))
    assert result.exit_code != 0
    assert "cannot overwrite" in result.output
    assert path.exists()
    result = _predict(path, "--output", str(BUNDLE / "model.json"))
    assert result.exit_code != 0
    assert "cannot overwrite" in result.output


def test_inspect_command_displays_safe_metadata():
    result = RUNNER.invoke(app, ["model", "inspect", "--model", str(BUNDLE), "--format", "json"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["training_n"] == 483
    assert info["training_events"] == 97
    assert info["feature_count"] == 20
    assert info["expression_type"] == EXPRESSION_TYPE
    assert info["status"] == "research_use_only"
    assert info["internal_cv"]["mean"] == pytest.approx(0.6825397634337843)
    assert "lacks independent compatible external validation" in info["external_validation"]
    assert "/Users/" not in result.stdout


def test_unsupported_orientation_zero_samples_and_duplicate_sample_ids(matrix, tmp_path):
    transposed = _write(matrix.T, tmp_path / "transposed.tsv")
    assert "unsupported orientation" in _predict(transposed).output
    zero = tmp_path / "zero.tsv"
    zero.write_text("gene\nABCB6\n")
    assert "zero sample columns" in _predict(zero).output
    duplicate = tmp_path / "duplicate-samples.tsv"
    duplicate.write_text("gene\tpatient\tpatient\nABCB6\t1\t2\n")
    assert "duplicate sample identifiers" in _predict(duplicate).output
