"""Cohort alignment, subsetting and analysis-set behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.errors import CohortAlignmentError, EndpointError
from hypoxiapipe.ingest.cohort import Cohort, Provenance, frame_checksum


def make_expr(n_genes: int = 20, n_samples: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic genes x samples matrix."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(8, 2, size=(n_genes, n_samples)),
        index=[f"G{i:03d}" for i in range(n_genes)],
        columns=[f"S{i:03d}" for i in range(n_samples)],
    )


def make_clinical(samples: list[str], n_missing: int = 0, seed: int = 1) -> pd.DataFrame:
    """Synthetic clinical table with a survival endpoint."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "time": rng.uniform(1, 60, size=len(samples)),
            "event": rng.integers(0, 2, size=len(samples)),
        },
        index=pd.Index(samples, name="sample_id"),
    )
    if n_missing:
        df.iloc[:n_missing, df.columns.get_loc("time")] = np.nan
    return df


PROV = Provenance(source="test")


def test_aligned_cohort_builds() -> None:
    expr = make_expr()
    clin = make_clinical(list(expr.columns))
    c = Cohort(name="T", expr=expr, clinical=clin, provenance=PROV)
    assert c.n_genes == 20
    assert c.n_samples == 40


def test_misaligned_samples_rejected() -> None:
    expr = make_expr()
    clin = make_clinical(list(expr.columns)[:-1])
    with pytest.raises(CohortAlignmentError, match="not aligned"):
        Cohort(name="T", expr=expr, clinical=clin, provenance=PROV)


def test_duplicate_sample_ids_rejected() -> None:
    expr = make_expr(n_samples=5)
    expr.columns = ["A", "B", "B", "C", "D"]
    clin = make_clinical(["A", "B", "C", "D", "E"])
    with pytest.raises(CohortAlignmentError, match="duplicate sample IDs"):
        Cohort(name="T", expr=expr, clinical=clin, provenance=PROV)


def test_align_takes_intersection_and_records_counts() -> None:
    expr = make_expr(n_samples=40)
    clin = make_clinical([*list(expr.columns)[:35], "EXTRA1", "EXTRA2"])
    c = Cohort.align("T", expr, clin, PROV, min_samples=30)
    assert c.n_samples == 35
    step = c.provenance.steps[-1]
    assert step.action == "align"
    assert step.detail["n_aligned"] == 35


def test_align_refuses_tiny_intersection() -> None:
    expr = make_expr(n_samples=40)
    clin = make_clinical(["NOPE1", "NOPE2"])
    with pytest.raises(CohortAlignmentError, match="only 0 samples align"):
        Cohort.align("T", expr, clin, PROV)


def test_analysis_set_drops_unusable_endpoints() -> None:
    expr = make_expr(n_samples=40)
    clin = make_clinical(list(expr.columns), n_missing=5)
    c = Cohort(name="T", expr=expr, clinical=clin, provenance=PROV)
    a = c.restrict_to_analysis_set("time", "event")
    assert a.n_samples == 35
    assert a.provenance.steps[-1].action == "analysis_set"
    assert "n_events" in a.provenance.steps[-1].detail


def test_analysis_set_changes_population_hash() -> None:
    """The population a z-score is relative to must be identifiable."""
    expr = make_expr(n_samples=40)
    clin = make_clinical(list(expr.columns), n_missing=5)
    c = Cohort(name="T", expr=expr, clinical=clin, provenance=PROV)
    assert c.population_hash != c.restrict_to_analysis_set("time", "event").population_hash


def test_analysis_set_before_scoring_changes_scores() -> None:
    """Standardise-then-subset and subset-then-standardise differ.

    This is the mechanism behind the unexplained hazard-ratio drift in the
    upstream project. The test pins the fact so the ordering is a decision.
    """
    from hypoxiapipe.scoring import score
    from hypoxiapipe.signatures.registry import Signature, compute_checksum

    expr = make_expr(n_genes=30, n_samples=60)
    genes = list(expr.index[:10])
    sig = Signature(
        name="test10",
        genes=tuple(genes),
        scoring="rowmean",
        checksum=compute_checksum(genes, None),
    )
    keep = list(expr.columns[:40])

    subset_then_score = score(expr.loc[:, keep], sig).scores
    score_then_subset = score(expr, sig).scores.loc[keep]

    assert not np.allclose(subset_then_score.to_numpy(), score_then_subset.to_numpy())


def test_missing_endpoint_column_raises() -> None:
    expr = make_expr()
    clin = make_clinical(list(expr.columns))
    c = Cohort(name="T", expr=expr, clinical=clin, provenance=PROV)
    with pytest.raises(EndpointError, match="no column 'bcr_time'"):
        c.restrict_to_analysis_set("bcr_time", "event")


def test_frame_checksum_is_order_invariant() -> None:
    expr = make_expr()
    shuffled = expr.iloc[::-1, ::-1]
    assert frame_checksum(expr) == frame_checksum(shuffled)


def test_frame_checksum_detects_a_single_changed_value() -> None:
    expr = make_expr()
    changed = expr.copy()
    changed.iloc[0, 0] += 1e-6
    assert frame_checksum(expr) != frame_checksum(changed)


def test_provenance_serialises() -> None:
    expr = make_expr()
    clin = make_clinical(list(expr.columns))
    c = Cohort.align("T", expr, clin, PROV).restrict_to_analysis_set("time", "event")
    d = c.summary()["provenance"]
    assert [s["action"] for s in d["steps"]] == ["align", "subset_samples", "analysis_set"]
