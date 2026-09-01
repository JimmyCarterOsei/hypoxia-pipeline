"""Golden-file regression tests against a verified real cohort.

These are the tests the rest of the suite cannot be: everything else runs on
synthetic fixtures and proves the code does what it was written to do. This
compares against numbers produced by a real, verified build — so it catches a
change that is internally consistent but wrong.

They are skipped unless a built cohort is available, because the data cannot be
committed:

    HYPOXIAPIPE_GOLDEN_COHORT=out/prad pytest tests/test_golden.py -v

Skipping is the honest default. A regression test that silently passes when it
cannot run is worse than one that says it did not run.

What is asserted, and what deliberately is not
----------------------------------------------
`population_hash` is the strongest value here: it covers the exact sample set
every score is relative to, and is independent of the expression values.

`expr_checksum` is **not** asserted. It moves whenever GDC re-releases the
underlying files, which is a data change rather than a regression, and a test
that fails on a legitimate upstream release trains people to ignore it.

Hazard ratios are compared with tolerances rather than exactly. The published
reference should be reproducible to well within 1%; requiring bit-identity
across BLAS versions and platforms would be false precision.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

GOLDEN = json.loads((Path(__file__).parent / "golden/tcga-prad.json").read_text())
COHORT_DIR = os.environ.get("HYPOXIAPIPE_GOLDEN_COHORT", "")

needs_cohort = pytest.mark.skipif(
    not COHORT_DIR or not Path(COHORT_DIR).exists(),
    reason="set HYPOXIAPIPE_GOLDEN_COHORT to a directory built by 'cohort build'",
)


@pytest.fixture(scope="module")
def cohort():
    """Load the built cohort, verifying its stored checksum."""
    from hypoxiapipe.ingest.store import load_cohort  # noqa: PLC0415

    return load_cohort(COHORT_DIR)


@pytest.fixture(scope="module")
def build_report():
    """Return the TCGA build report written alongside the cohort."""
    meta = json.loads((Path(COHORT_DIR) / "cohort.json").read_text())
    return meta["build"]["tcga"]


# --------------------------------------------------------------------------
# the reference itself
# --------------------------------------------------------------------------


def test_the_golden_file_records_its_provenance():
    # A frozen number without the release it came from cannot be re-derived.
    assert GOLDEN["gdc_data_release"]
    assert GOLDEN["recorded"]
    assert GOLDEN["spec"] == "tcga-prad"


def test_the_pinned_spec_agrees_with_the_golden_file():
    from hypoxiapipe.ingest.spec import load_bundled_cohort  # noqa: PLC0415

    spec = load_bundled_cohort("tcga-prad")
    assert spec.expect.n_samples == GOLDEN["cohort"]["n_samples"], (
        "the spec and the golden reference disagree about the cohort size"
    )
    assert spec.cdr_endpoint == GOLDEN["build"]["endpoint"]["endpoint"]


# --------------------------------------------------------------------------
# cohort assembly
# --------------------------------------------------------------------------


@needs_cohort
def test_sample_set_is_unchanged(cohort):
    # The hash covers the exact population every score is relative to.
    assert cohort.population_hash == GOLDEN["cohort"]["population_hash"]
    assert cohort.n_samples == GOLDEN["cohort"]["n_samples"]


@needs_cohort
def test_the_counts_reconcile_at_every_stage(build_report):
    expected = GOLDEN["build"]
    assert build_report["manifest"]["n_files"] == expected["manifest"]["n_files"]
    assert (
        build_report["selection"]["n_dropped_non_primary"]
        == expected["selection"]["n_dropped_non_primary"]
    )
    assert (
        build_report["selection"]["n_patients_with_multiple_aliquots"]
        == expected["selection"]["n_patients_with_multiple_aliquots"]
    )
    assert build_report["join"]["n_joined"] == expected["join"]["n_joined"]
    # Every expression patient must have clinical data; an orphan means the
    # barcode truncation or the CDR subset changed.
    assert build_report["join"]["n_expression_without_clinical"] == 0


@needs_cohort
def test_the_endpoint_is_unchanged(build_report):
    expected = GOLDEN["build"]["endpoint"]
    for key in ("endpoint", "n_events", "cap_months", "n_censored_by_cap", "n_project"):
        assert build_report["endpoint"][key] == expected[key], f"{key} moved"


@needs_cohort
def test_the_matrix_is_log2_transformed_exactly_once(cohort):
    # The regression that shipped: log2 applied by both the loader and the
    # pipeline, capping the range near 4 instead of ~15.
    observed = float(cohort.expr.to_numpy().max())
    assert GOLDEN["cohort"]["scale_max_min"] <= observed <= GOLDEN["cohort"]["scale_max_max"], (
        f"expression maximum is {observed:.2f}; a value near 4 means a double log2"
    )


# --------------------------------------------------------------------------
# scoring and survival
# --------------------------------------------------------------------------


@needs_cohort
@pytest.mark.parametrize("name", sorted(GOLDEN["coverage"]))
def test_signature_coverage_is_unchanged(cohort, name):
    from hypoxiapipe.scoring import score  # noqa: PLC0415
    from hypoxiapipe.signatures.registry import load_bundled  # noqa: PLC0415

    result = score(cohort.expr, load_bundled(name))
    expected = GOLDEN["coverage"][name]
    assert result.n_found == expected["n_found"]
    assert result.n_total == expected["n_total"]


@needs_cohort
@pytest.mark.parametrize(
    ("name", "action"),
    [(n, a) for n, actions in GOLDEN["survival"].items() for a in actions],
)
def test_survival_estimates_reproduce(cohort, name, action):
    from hypoxiapipe.modeling import cox, r_available  # noqa: PLC0415
    from hypoxiapipe.scoring import score  # noqa: PLC0415
    from hypoxiapipe.signatures.registry import load_bundled  # noqa: PLC0415

    if not r_available():
        pytest.skip("R is required for survival estimation")

    expected = GOLDEN["survival"][name][action]
    scores = score(cohort.expr, load_bundled(name)).scores
    (result,) = cox(
        cohort.clinical["time_months"], cohort.clinical["event"], {name: scores}, action=action
    )
    tol = GOLDEN["tolerances"]

    assert result.n == expected["n"]
    assert result.n_events == expected["n_events"]
    assert result.hr == pytest.approx(expected["hr"], rel=tol["hr_rel"])
    assert result.ci_low == pytest.approx(expected["ci_low"], rel=tol["hr_rel"])
    assert result.ci_high == pytest.approx(expected["ci_high"], rel=tol["hr_rel"])
    assert result.c_index == pytest.approx(expected["c_index"], abs=tol["c_index_abs"])
    # p-values span orders of magnitude, so compare on a log scale.
    assert math.log10(result.p) == pytest.approx(math.log10(expected["p"]), abs=tol["p_log10_abs"])
