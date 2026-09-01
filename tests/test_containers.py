"""Phase 6 tests: the R transport, and what the images promise.

Docker is not available in CI's test job, so the container path is exercised by
pointing ``HYPOXIAPIPE_R_COMMAND`` at a stand-in that speaks the same contract.
That is not a weaker test than running Docker: the contract *is* one JSON
request on stdin and one response on stdout, so anything honouring it is a
valid transport. The images themselves are built and smoke-tested in
``.github/workflows/images.yml``, which is where a Docker daemon exists.
"""

from __future__ import annotations

import json
import re
import stat
import sys
from pathlib import Path

import pytest

from hypoxiapipe.modeling import rbridge
from hypoxiapipe.modeling.rbridge import (
    R_COMMAND_ENV,
    RBridgeError,
    call_r,
    cox,
    r_available,
    r_command,
    worker_script,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    """Install a stand-in worker that speaks the JSON contract, and return it."""
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "n = len(request['time'])\n"
        "print(json.dumps({'ok': True, 'action': request['action'],\n"
        "  'r_version': 'fake', 'results': [\n"
        "    {'name': name, 'model': 'per_sd', 'term': 'z', 'n': n,\n"
        "     'n_events': int(sum(request['event'])), 'hr': 1.5, 'ci_low': 1.1,\n"
        "     'ci_high': 2.0, 'p': 0.001, 'c_index': 0.62}\n"
        "    for name in request['scores']]}))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(R_COMMAND_ENV, f"{sys.executable} {script}")
    return script


# --------------------------------------------------------------------------
# transport resolution
# --------------------------------------------------------------------------


def test_default_transport_runs_the_bundled_worker(monkeypatch):
    monkeypatch.delenv(R_COMMAND_ENV, raising=False)
    argv = r_command()
    if argv is None:
        pytest.skip("no local Rscript")
    assert argv[0].endswith("Rscript")
    assert argv[-1] == str(worker_script())


def test_an_override_replaces_the_command_entirely(monkeypatch):
    monkeypatch.setenv(R_COMMAND_ENV, "docker run --rm -i ghcr.io/acme/hypoxiapipe-r:1.2.3")
    argv = r_command()
    assert argv == ["docker", "run", "--rm", "-i", "ghcr.io/acme/hypoxiapipe-r:1.2.3"]
    # The image's entrypoint is the worker, so no script path is appended -
    # doing so would pass a host path into a container that cannot see it.
    assert not any("survival_worker" in part for part in argv)


def test_override_is_split_as_argv_never_run_through_a_shell(monkeypatch):
    monkeypatch.setenv(R_COMMAND_ENV, "docker run --rm -i 'my image:tag'")
    assert r_command() == ["docker", "run", "--rm", "-i", "my image:tag"]


def test_blank_override_falls_back_to_local_r(monkeypatch):
    monkeypatch.setenv(R_COMMAND_ENV, "   ")
    argv = r_command()
    assert argv is None or argv[0].endswith("Rscript")


def test_a_configured_transport_counts_as_available(monkeypatch):
    monkeypatch.setenv(R_COMMAND_ENV, "docker run --rm -i example:latest")
    # Taken at its word: probing would start a container on every import.
    assert r_available()


def test_no_r_anywhere_is_an_actionable_error(monkeypatch):
    monkeypatch.delenv(R_COMMAND_ENV, raising=False)
    monkeypatch.setattr(rbridge, "rscript_path", lambda: None)
    assert r_command() is None
    with pytest.raises(RBridgeError, match=R_COMMAND_ENV):
        call_r({"action": "cox_persd", "time": [1.0], "event": [1], "scores": {}})


def test_an_unstartable_transport_reports_the_command(monkeypatch):
    monkeypatch.setenv(R_COMMAND_ENV, "/nonexistent/binary --flag")
    with pytest.raises(RBridgeError, match="could not start the R worker"):
        call_r({"action": "cox_persd", "time": [1.0], "event": [1], "scores": {}})


# --------------------------------------------------------------------------
# the transport actually carries a request
# --------------------------------------------------------------------------


def test_a_stand_in_transport_completes_a_round_trip(fake_worker):
    results = cox([10.0] * 30, [1, 0] * 15, {"sig": list(range(30))})
    assert len(results) == 1
    assert results[0].n == 30
    assert results[0].hr == 1.5


def test_a_transport_that_returns_junk_is_caught(tmp_path, monkeypatch):
    script = tmp_path / "junk.py"
    script.write_text("print('not json at all')\n")
    monkeypatch.setenv(R_COMMAND_ENV, f"{sys.executable} {script}")
    with pytest.raises(RBridgeError, match="unparseable output"):
        cox([10.0] * 20, [1] * 20, {"sig": list(range(20))})


def test_a_silent_transport_is_caught(tmp_path, monkeypatch):
    script = tmp_path / "silent.py"
    script.write_text("import sys; sys.exit(3)\n")
    monkeypatch.setenv(R_COMMAND_ENV, f"{sys.executable} {script}")
    with pytest.raises(RBridgeError, match="no output"):
        cox([10.0] * 20, [1] * 20, {"sig": list(range(20))})


# --------------------------------------------------------------------------
# the request fixture the image smoke tests rely on
# --------------------------------------------------------------------------


def test_the_smoke_test_fixture_is_a_valid_request():
    request = json.loads((REPO_ROOT / "tests/fixtures/r_request.json").read_text())
    assert request["action"] == "cox_persd"
    assert len(request["time"]) == len(request["event"]) == 120
    assert set(request["event"]) <= {0, 1}
    assert all(t > 0 for t in request["time"])
    for values in request["scores"].values():
        assert len(values) == 120


# --------------------------------------------------------------------------
# image definitions
# --------------------------------------------------------------------------


def test_both_images_exist_and_are_separate():
    python_image = (DOCKER_DIR / "Dockerfile.python").read_text()
    r_image = (DOCKER_DIR / "Dockerfile.r").read_text()
    # One fat container is the trap: the Python image must not install R, and
    # the R image must not install the Python package.
    assert "r-base" not in python_image and "rocker" not in python_image
    assert "pip install" not in r_image


def test_images_pin_their_base_versions():
    for name in ("Dockerfile.python", "Dockerfile.r"):
        for line in (DOCKER_DIR / name).read_text().splitlines():
            if line.startswith("FROM "):
                base = line.split()[1]
                assert ":" in base, f"{name}: unpinned base image {base!r}"
                assert not base.endswith(":latest"), f"{name}: base pinned to :latest"


def test_images_run_as_a_non_root_user():
    for name in ("Dockerfile.python", "Dockerfile.r"):
        text = (DOCKER_DIR / name).read_text()
        users = re.findall(r"^USER\s+(\S+)", text, flags=re.MULTILINE)
        assert users, f"{name}: no USER directive"
        assert users[-1] != "root", f"{name}: final USER is root"


def test_the_r_image_entrypoint_is_the_worker():
    text = (DOCKER_DIR / "Dockerfile.r").read_text()
    assert "survival_worker.R" in text
    entrypoint = re.search(r"^ENTRYPOINT\s+(.+)$", text, flags=re.MULTILINE)
    assert entrypoint and "survival_worker.R" in entrypoint.group(1)


def test_dockerignore_keeps_data_and_git_out_of_the_context():
    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    for pattern in (".git", "*.parquet", "data"):
        assert pattern in ignored, f"{pattern} should be excluded from the build context"
