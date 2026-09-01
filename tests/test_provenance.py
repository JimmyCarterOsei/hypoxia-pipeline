"""Phase 3 tests: content hashing and run manifests."""

from __future__ import annotations

import json

import pytest

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.ingest.pipeline import build_cohort
from hypoxiapipe.ingest.store import save_cohort
from hypoxiapipe.provenance import (
    MANIFEST_FILE,
    RunManifest,
    environment,
    hash_bytes,
    hash_directory,
    hash_file,
    hash_json,
    hash_text,
    verify_manifest,
)
from hypoxiapipe.signatures.registry import load_bundled

# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def test_digests_are_tagged_so_they_cannot_be_confused_with_bare_hex():
    assert hash_text("x").startswith("sha256:")
    assert hash_bytes(b"x") == hash_text("x")


def test_file_hash_matches_content_hash(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert hash_file(p) == hash_text("hello")


def test_json_hash_ignores_key_order_but_not_values():
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})
    assert hash_json({"a": 1}) != hash_json({"a": 2})


def test_hash_directory_maps_relative_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    hashes = hash_directory(tmp_path)
    assert set(hashes) == {"a.txt", "sub/b.txt"}
    assert hashes["a.txt"] == hash_text("a")


# --------------------------------------------------------------------------
# manifest content
# --------------------------------------------------------------------------


def test_environment_records_code_version_and_git_state():
    env = environment()
    assert env["hypoxiapipe_version"]
    assert env["python"]
    assert "git_sha" in env  # None outside a repo is a legitimate recorded value


def test_manifest_records_signatures_by_checksum():
    manifest = RunManifest()
    sig = load_bundled("smith20")
    manifest.absorb_signatures([sig])
    recorded = manifest.inputs[0]
    assert recorded.kind == "signature"
    assert recorded.checksum == sig.checksum
    assert recorded.detail["n_genes"] == sig.n_genes


def test_manifest_hash_changes_with_parameters_but_not_with_outputs(tmp_path):
    a = RunManifest(run_id="fixed", started_at="2026-01-01T00:00:00+00:00", params={"k": 1})
    b = RunManifest(run_id="fixed", started_at="2026-01-01T00:00:00+00:00", params={"k": 2})
    assert a.to_dict()["manifest_hash"] != b.to_dict()["manifest_hash"]

    before = a.to_dict()["manifest_hash"]
    written = tmp_path / "out.txt"
    written.write_text("result")
    a.add_output_file(written)
    # outputs accumulate as a run proceeds, so they are outside the self-hash
    assert a.to_dict()["manifest_hash"] == before


def test_manifest_round_trips_through_disk(tmp_path):
    manifest = RunManifest(params={"spec": "demo"})
    manifest.add_input("GSE00001", checksum=hash_text("m"), kind="cohort", n_samples=48)
    manifest.add_step("download", url="fixture://x")
    manifest.close(command="cohort build demo")
    path = manifest.write(tmp_path)

    assert path.name == MANIFEST_FILE
    loaded = json.loads(path.read_text())
    assert loaded["command"] == "cohort build demo"
    assert loaded["finished_at"]
    assert loaded["inputs"][0]["detail"]["n_samples"] == 48
    assert loaded["steps"][0]["action"] == "download"
    assert RunManifest.load(tmp_path)["run_id"] == manifest.run_id


def test_loading_a_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(HypoxiapipeError, match="no manifest"):
        RunManifest.load(tmp_path)


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _manifest_with_output(tmp_path):
    written = tmp_path / "result.tsv"
    written.write_text("gene\tscore\nALDOA\t1.2\n")
    manifest = RunManifest()
    manifest.add_output_file(written)
    manifest.close()
    manifest.write(tmp_path)
    return written


def test_verify_passes_on_untouched_outputs(tmp_path):
    _manifest_with_output(tmp_path)
    result = verify_manifest(tmp_path)
    assert result.ok
    assert result.checked == 1
    assert "PASS" in result.to_markdown()


def test_verify_detects_an_edited_output(tmp_path):
    written = _manifest_with_output(tmp_path)
    written.write_text("gene\tscore\nALDOA\t9.9\n")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert result.changed and not result.missing
    assert "CHANGED" in result.to_markdown()


def test_verify_detects_a_deleted_output(tmp_path):
    written = _manifest_with_output(tmp_path)
    written.unlink()
    result = verify_manifest(tmp_path)
    assert result.missing
    assert not result.ok


def test_verify_relocates_relative_paths_with_base(tmp_path):
    written = _manifest_with_output(tmp_path)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / written.name).write_text(written.read_text())
    written.unlink()
    # the manifest stays put; only the artefacts moved
    assert verify_manifest(tmp_path, base=moved).ok


# --------------------------------------------------------------------------
# integration with a cohort build
# --------------------------------------------------------------------------


def test_cohort_build_manifest_records_inputs_steps_and_outputs(tmp_path, primed_cache, geo_spec):
    """A built cohort's manifest should identify every input by content hash."""
    sig = load_bundled("smith20")
    result = build_cohort(geo_spec, primed_cache, signatures=[sig], min_samples=10)

    out = tmp_path / "built"
    save_cohort(result.cohort, out)

    manifest = RunManifest(params={"spec": "TestCohort"})
    manifest.absorb_cohort(result)
    manifest.absorb_signatures([sig])
    manifest.add_output_file(out / "expression.parquet")
    manifest.close(command="cohort build")
    manifest.write(out)

    kinds = {a.kind for a in manifest.inputs}
    assert {"cohort", "signature", "symbol_authority"} <= kinds

    cohort_input = next(a for a in manifest.inputs if a.kind == "cohort")
    assert cohort_input.checksum == result.cohort.expr_checksum
    assert cohort_input.detail["population_hash"] == result.cohort.population_hash

    actions = [s["action"] for s in manifest.steps]
    assert "map_probes_to_symbols" in actions
    assert "harmonise_symbols" in actions

    assert verify_manifest(out).ok


def test_the_offline_guard_is_actually_armed():
    """The no-network fixture must bite; otherwise the offline claim is decorative."""
    import socket  # noqa: PLC0415

    with pytest.raises(RuntimeError, match="attempted a network connection"):
        socket.create_connection(("ftp.ncbi.nlm.nih.gov", 443), timeout=1)
