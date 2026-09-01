"""Golden-file regression tests against verified real builds.

These are the tests the rest of the suite cannot be: everything else runs on
synthetic fixtures and proves the code does what it was written to do. These
compare against numbers from real, verified builds — so they catch a change
that is internally consistent but wrong.

Point the suite at a directory of built cohorts:

    hypoxiapipe cohort build ... --out out/prad
    HYPOXIAPIPE_GOLDEN_DIR=out pytest tests/test_golden.py -v

Each `tests/golden/<name>.json` names the subdirectory it expects under that
directory. Cohorts that are absent skip individually, so a partial set still
checks what it can. Skipping is the honest default: a regression test that
silently passes when it cannot run is worse than one that says it did not run.

What is asserted, and what deliberately is not
----------------------------------------------
`population_hash` is the strongest value available: it covers the exact sample
set every score is relative to, independent of the expression values.

`expr_checksum` is **not** asserted. It moves whenever the source re-deposits
its data, which is a data change rather than a regression, and a test that
fails on a legitimate upstream release trains people to ignore it.

Hazard ratios are compared with tolerances. Requiring bit-identity across BLAS
versions and platforms would be false precision.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN = {p.stem: json.loads(p.read_text()) for p in sorted(GOLDEN_DIR.glob("*.json"))}

#: Directory holding built cohorts, one subdirectory per cohort.
BUILD_DIR = os.environ.get("HYPOXIAPIPE_GOLDEN_DIR", "")

#: Where each golden reference expects to find its build.
COHORT_DIRS = {"tcga-prad": "prad", "cambridge": "cambridge", "stockholm": "stockholm"}

NAMES = sorted(GOLDEN)
SURVIVAL_CASES = [
    (name, sig, action)
    for name, ref in sorted(GOLDEN.items())
    for sig, actions in ref.get("survival", {}).items()
    for action in actions
]


def cohort_path(name: str) -> Path | None:
    """Return the built cohort directory for a golden reference, if present."""
    if not BUILD_DIR:
        # Back-compatible single-cohort form.
        legacy = os.environ.get("HYPOXIAPIPE_GOLDEN_COHORT", "")
        if legacy and name == "tcga-prad" and Path(legacy).exists():
            return Path(legacy)
        return None
    candidate = Path(BUILD_DIR) / COHORT_DIRS.get(name, name)
    return candidate if candidate.exists() else None


def load(name: str):
    """Load a built cohort, or skip when it is not available."""
    path = cohort_path(name)
    if path is None:
        pytest.skip(f"no build for '{name}'; set HYPOXIAPIPE_GOLDEN_DIR")
    from hypoxiapipe.ingest.store import load_cohort  # noqa: PLC0415

    return load_cohort(path)


# --------------------------------------------------------------------------
# the references themselves (always run)
# --------------------------------------------------------------------------


def test_there_are_references_for_every_bundled_cohort():
    from hypoxiapipe.ingest.spec import list_bundled_cohorts  # noqa: PLC0415

    assert set(GOLDEN) == set(list_bundled_cohorts()), (
        "every bundled cohort should have a frozen reference once it has been built"
    )


@pytest.mark.parametrize("name", NAMES)
def test_each_reference_records_its_provenance(name):
    ref = GOLDEN[name]
    # A frozen number without the release it came from cannot be re-derived.
    assert ref["recorded"]
    assert ref.get("gdc_data_release") or ref.get("source")


@pytest.mark.parametrize("name", NAMES)
def test_each_reference_agrees_with_its_pinned_spec(name):
    from hypoxiapipe.ingest.spec import load_bundled_cohort  # noqa: PLC0415

    spec = load_bundled_cohort(name)
    assert spec.expect.n_samples == GOLDEN[name]["cohort"]["n_samples"], (
        f"{name}: the spec and its golden reference disagree about cohort size"
    )


@pytest.mark.parametrize("name", NAMES)
def test_no_reference_records_an_implausible_event_rate(name):
    # A rate near 1.0 means censored patients were dropped rather than
    # censored; freezing such a number would enshrine the bug.
    rate = GOLDEN[name]["cohort"].get("event_rate")
    if rate is not None:
        assert rate < 0.85, f"{name}: {rate:.0%} event rate - censored patients dropped?"


def test_the_signature_replicates_out_of_sample():
    """The point of the whole exercise: a consistent effect outside discovery.

    smith20 was derived in TCGA-PRAD, so that estimate is in-sample. Cambridge
    and Stockholm are independent, on a different platform with a different
    endpoint definition. This asserts direction and rough magnitude agree, not
    that the numbers match - they should not.
    """
    external = [n for n in NAMES if n != "tcga-prad"]
    assert external, "no external cohorts to validate against"

    for name in external:
        result = GOLDEN[name]["survival"]["smith20"]["cox_persd"]
        assert result["hr"] > 1.0, f"{name}: effect reversed out of sample"
        assert result["ci_low"] > 1.0, f"{name}: effect not significant out of sample"
        assert result["c_index"] > 0.55, f"{name}: no discrimination out of sample"

    discovery = GOLDEN["tcga-prad"]["survival"]["smith20"]["cox_persd"]["hr"]
    for name in external:
        hr = GOLDEN[name]["survival"]["smith20"]["cox_persd"]["hr"]
        # Within a factor of two of discovery: loose on purpose, since some
        # shrinkage out of sample is expected and healthy.
        assert 0.5 < hr / discovery < 2.0, f"{name}: HR {hr} far from discovery {discovery}"


# --------------------------------------------------------------------------
# built cohorts (skip individually when absent)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_sample_set_is_unchanged(name):
    cohort = load(name)
    expected = GOLDEN[name]["cohort"]
    assert cohort.population_hash == expected["population_hash"]
    assert cohort.n_samples == expected["n_samples"]


@pytest.mark.parametrize("name", NAMES)
def test_event_count_is_unchanged(name):
    cohort = load(name)
    assert int(cohort.clinical["event"].sum()) == GOLDEN[name]["cohort"]["n_events"]


@pytest.mark.parametrize("name", NAMES)
def test_the_matrix_is_log2_transformed_exactly_once(name):
    # The regression that shipped: log2 applied by both the TCGA loader and the
    # pipeline, capping the range near 4 instead of ~15.
    cohort = load(name)
    expected = GOLDEN[name]["cohort"]
    observed = float(cohort.expr.to_numpy().max())
    assert expected["scale_max_min"] <= observed <= expected["scale_max_max"], (
        f"{name}: expression maximum is {observed:.2f}; a value near 4 means a double log2"
    )


@pytest.mark.parametrize("name", NAMES)
def test_signature_coverage_is_unchanged(name):
    from hypoxiapipe.scoring import score  # noqa: PLC0415
    from hypoxiapipe.signatures.registry import load_bundled  # noqa: PLC0415

    cohort = load(name)
    for sig_name, expected in GOLDEN[name]["coverage"].items():
        result = score(cohort.expr, load_bundled(sig_name))
        assert result.n_found == expected["n_found"], sig_name
        assert result.n_total == expected["n_total"], sig_name


@pytest.mark.parametrize(("name", "sig_name", "action"), SURVIVAL_CASES)
def test_survival_estimates_reproduce(name, sig_name, action):
    from hypoxiapipe.modeling import cox, r_available  # noqa: PLC0415
    from hypoxiapipe.scoring import score  # noqa: PLC0415
    from hypoxiapipe.signatures.registry import load_bundled  # noqa: PLC0415

    cohort = load(name)
    if not r_available():
        pytest.skip("R is required for survival estimation")

    expected = GOLDEN[name]["survival"][sig_name][action]
    scores = score(cohort.expr, load_bundled(sig_name)).scores
    (result,) = cox(
        cohort.clinical["time_months"],
        cohort.clinical["event"],
        {sig_name: scores},
        action=action,
    )
    tol = GOLDEN[name]["tolerances"]

    assert result.n == expected["n"]
    assert result.n_events == expected["n_events"]
    assert result.hr == pytest.approx(expected["hr"], rel=tol["hr_rel"])
    assert result.ci_low == pytest.approx(expected["ci_low"], rel=tol["hr_rel"])
    assert result.ci_high == pytest.approx(expected["ci_high"], rel=tol["hr_rel"])
    assert result.c_index == pytest.approx(expected["c_index"], abs=tol["c_index_abs"])
    # p-values span orders of magnitude, so compare on a log scale.
    assert math.log10(result.p) == pytest.approx(math.log10(expected["p"]), abs=tol["p_log10_abs"])
