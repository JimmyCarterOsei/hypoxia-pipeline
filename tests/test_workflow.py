"""Phase 7 tests: the Nextflow workflow definition.

Nextflow is not available in the unit-test job, so these check the properties
of the workflow that can be checked statically — that processes declare
containers and resources, that the test profile is genuinely offline, that the
fixtures it depends on exist. The workflow is *executed* by the `workflow` job
in ``.github/workflows/ci.yml``, which installs Nextflow and runs
``-profile test`` end to end; that is what proves it works.

Static checks like these are worth having anyway: they catch the class of
mistake where someone adds a process and forgets the container directive, which
runs fine locally and then fails on AWS Batch two phases later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "workflow"
MAIN_NF = WORKFLOW_DIR / "main.nf"
PROCESSES_NF = WORKFLOW_DIR / "modules/processes.nf"
CONFIG = WORKFLOW_DIR / "nextflow.config"
FIXTURES = REPO_ROOT / "tests/fixtures/workflow"

EXPECTED_PROCESSES = {
    "BUILD_COHORT",
    "VALIDATE_SIGNATURE",
    "CROSS_VALIDATE",
    "VERIFY_MANIFESTS",
    "COLLECT_RESULTS",
}


@pytest.fixture(scope="module")
def processes() -> dict[str, str]:
    """Return each process body keyed by name."""
    text = PROCESSES_NF.read_text()
    blocks: dict[str, str] = {}
    starts = [(m.group(1), m.start()) for m in re.finditer(r"^process\s+(\w+)\s*\{", text, re.M)]
    for i, (name, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        blocks[name] = text[start:end]
    return blocks


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_workflow_files_exist():
    for path in (MAIN_NF, PROCESSES_NF, CONFIG):
        assert path.exists(), f"missing {path}"


def test_all_processes_are_defined_and_imported(processes):
    assert set(processes) == EXPECTED_PROCESSES
    included = MAIN_NF.read_text()
    for name in EXPECTED_PROCESSES:
        assert name in included, f"{name} is defined but never included in main.nf"


def test_dsl2_is_enabled():
    for path in (MAIN_NF, PROCESSES_NF):
        assert "nextflow.enable.dsl = 2" in path.read_text()


def test_every_process_declares_a_container(processes):
    # A process without one runs in whatever the host happens to have, which
    # works locally and then fails on a batch executor.
    for name, body in processes.items():
        assert "container " in body, f"{name} has no container directive"


def test_every_process_declares_resources(processes):
    for name, body in processes.items():
        assert re.search(r"label\s+'process_(low|medium|high)'", body), (
            f"{name} has no resource label; AWS Batch needs cpus/memory to schedule"
        )


def test_resource_labels_are_all_configured(processes):
    config = CONFIG.read_text()
    used = set(re.findall(r"label\s+'(process_\w+)'", PROCESSES_NF.read_text()))
    for label in used:
        assert f"withLabel: {label}" in config, f"{label} is used but not configured"


def test_published_outputs_are_copied_not_symlinked(processes):
    # Symlinks into work/ break the moment the scratch directory is cleaned.
    for name, body in processes.items():
        if "publishDir" in body:
            assert "mode: 'copy'" in body, f"{name} publishes without copying"


# --------------------------------------------------------------------------
# the test profile
# --------------------------------------------------------------------------


def test_test_profile_is_offline_and_needs_no_containers():
    config = CONFIG.read_text()
    profile = config[config.index("    test {") :]
    profile = profile[: profile.index("\n    }")]
    assert "params.offline    = true" in profile.replace("=true", "= true")
    assert "docker.enabled   = false" in profile.replace("=false", "= false")


def test_test_profile_fixtures_exist(workflow_fixtures):
    spec = workflow_fixtures / "demo_cohort.yaml"
    assert spec.exists()
    assert (workflow_fixtures / "cache").is_dir()

    parsed = yaml.safe_load(spec.read_text())
    assert parsed["source"] == "geo"
    assert parsed["expect"]["n_samples"] > 0


def test_test_profile_fixture_cache_has_what_the_spec_asks_for(workflow_fixtures):
    spec = yaml.safe_load((workflow_fixtures / "demo_cohort.yaml").read_text())
    cached = {p.name for p in (workflow_fixtures / "cache").rglob("*") if p.is_file()}
    assert any(spec["accession"] in name for name in cached), "series matrix not cached"
    assert any(spec["platform"] in name for name in cached), "platform annotation not cached"


def test_the_fixture_spec_pins_its_expectations(workflow_fixtures):
    # The fixture is generated, so its sample count is known exactly; pinning
    # it means a change to the generator fails the smoke run loudly.
    spec = yaml.safe_load((workflow_fixtures / "demo_cohort.yaml").read_text())
    assert spec["expect"]["n_samples"] == 140


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_profiles_cover_local_containers_and_batch():
    config = CONFIG.read_text()
    for profile in ("standard", "docker", "singularity", "test", "awsbatch"):
        assert f"    {profile} {{" in config, f"no '{profile}' profile"


def test_container_profiles_configure_the_r_transport():
    config = CONFIG.read_text()
    for profile in ("docker", "singularity"):
        block = config[config.index(f"    {profile} {{") :]
        block = block[: block.index("\n    }")]
        assert "HYPOXIAPIPE_R_COMMAND" in block, (
            f"{profile} runs the Python image but does not tell it how to reach R"
        )


def test_failures_stop_the_run_by_default():
    config = CONFIG.read_text()
    assert "errorStrategy = 'terminate'" in config
    assert "maxRetries    = 0" in config


def test_run_provenance_is_captured():
    config = CONFIG.read_text()
    for block in ("timeline", "report", "trace", "dag"):
        assert re.search(rf"^{block}\s*\{{", config, re.M), f"no {block} block"
        assert "overwrite = true" in config


def test_images_are_declared_as_parameters():
    config = CONFIG.read_text()
    assert "python_image" in config
    assert "r_image" in config
