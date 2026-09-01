"""Symbol harmonisation against a pinned authority, and cohort QC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.errors import AliasTableError, QCFailure
from hypoxiapipe.harmonise import aliases, symbols
from hypoxiapipe.ingest.cohort import Cohort, Provenance
from hypoxiapipe.qc import Level, infer_scale, run_qc
from hypoxiapipe.signatures import registry

PROV = Provenance(source="test")


def matrix(index: list[str], n_samples: int = 40, seed: int = 3) -> pd.DataFrame:
    """Synthetic matrix with the given row labels."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(8, 1.5, size=(len(index), n_samples)),
        index=index,
        columns=[f"S{i:03d}" for i in range(n_samples)],
    )


# ------------------------------------------------------------------ alias table


def test_bundled_table_matches_its_pinned_checksum() -> None:
    """The pin is the point: an edited mapping table must not load silently."""
    table = aliases.load_table(verify=True)
    assert table.checksum == aliases.BUNDLED_CHECKSUMS[aliases.DEFAULT_RELEASE]


def test_bundled_table_declares_itself_a_subset() -> None:
    """A partial authority presented as complete would be the error we prevent."""
    table = aliases.load_table()
    assert table.is_subset
    assert "subset" in table.authority


def test_edited_table_fails_verification(tmp_path: Path) -> None:
    table = aliases.load_table()
    assert aliases.checksum_text("alias\tapproved\nFOO\tBAR\n") != table.checksum


def test_unknown_release_raises() -> None:
    with pytest.raises(AliasTableError, match="no bundled alias table"):
        aliases.load_table(release="1999-01-01")


def test_external_table_loads_and_hashes(tmp_path: Path) -> None:
    p = tmp_path / "custom.tsv"
    p.write_text("alias\tapproved\nOLD1\tNEW1\nOLD2\tNEW2\n")
    table = aliases.load_table(path=p)
    assert table.n_entries == 2
    assert table.checksum.startswith("sha256:")
    assert table.approved_for("old1") == ("NEW1",)


# --------------------------------------------------------------- harmonisation


def test_known_alias_is_remapped() -> None:
    expr = matrix(["AK3L1", "ALDOA", "VEGFA"])
    out, rep = symbols.harmonise_symbols(expr, aliases.load_table())
    assert "AK4" in out.index
    assert "AK3L1" not in out.index
    assert rep.remapped == {"AK3L1": "AK4"}


def test_current_symbols_are_untouched() -> None:
    expr = matrix(["ALDOA", "VEGFA", "NDRG1"])
    out, rep = symbols.harmonise_symbols(expr, aliases.load_table())
    assert list(out.index) == ["ALDOA", "VEGFA", "NDRG1"]
    assert rep.remapped == {}


def test_ambiguous_alias_is_reported_not_guessed(tmp_path: Path) -> None:
    """One alias, two approved targets: keep the original and say so."""
    p = tmp_path / "ambig.tsv"
    p.write_text("alias\tapproved\nDEC1\tBHLHE40\nDEC1\tDEC1\n")
    table = aliases.load_table(path=p)
    out, rep = symbols.harmonise_symbols(matrix(["DEC1", "ALDOA"]), table)
    assert "DEC1" in out.index
    assert rep.ambiguous["DEC1"] == ("BHLHE40", "DEC1")
    assert rep.remapped == {}


def test_remapping_that_creates_a_duplicate_is_collapsed() -> None:
    """CYR61 -> CCN1 when CCN1 is already present must not yield two CCN1 rows."""
    expr = matrix(["CYR61", "CCN1", "ALDOA"])
    expr.loc["CCN1"] = expr.loc["CCN1"] + 5  # higher mean: max_mean should keep it
    out, rep = symbols.harmonise_symbols(expr, aliases.load_table())
    assert out.index.is_unique
    assert rep.collapsed["CCN1"] == 2
    assert out.loc["CCN1"].mean() > 10


def test_collapse_rules_differ_and_are_explicit() -> None:
    expr = pd.DataFrame([[1.0, 1.0], [9.0, 9.0]], index=["G", "G"], columns=["a", "b"])
    mean_out, _ = symbols.collapse_duplicate_rows(expr, rule="mean")
    max_out, _ = symbols.collapse_duplicate_rows(expr, rule="max_mean")
    first_out, _ = symbols.collapse_duplicate_rows(expr, rule="first")
    assert mean_out.loc["G", "a"] == 5.0
    assert max_out.loc["G", "a"] == 9.0
    assert first_out.loc["G", "a"] == 1.0


def test_unknown_collapse_rule_rejected() -> None:
    with pytest.raises(ValueError, match="unknown collapse rule"):
        symbols.collapse_duplicate_rows(matrix(["A", "A"]), rule="whatever")


def test_signature_symbols_are_reported_not_rewritten() -> None:
    """Yang lists CYR61; the current approved symbol is CCN1.

    The spec is hashed against its published source, so the fix belongs on the
    matrix side, not by quietly editing a verified gene list.
    """
    yang = registry.load_bundled("yang28")
    outdated = symbols.check_signature_symbols(yang, aliases.load_table())
    assert outdated.get("CYR61") == "CCN1"
    assert "CYR61" in yang.genes  # unchanged


def test_symbol_report_serialises() -> None:
    _, rep = symbols.harmonise_symbols(matrix(["AK3L1", "ALDOA"]), aliases.load_table())
    d = rep.to_dict()
    assert d["n_remapped"] == 1
    assert d["collapse_rule"] == "max_mean"
    assert d["authority_checksum"].startswith("sha256:")


# -------------------------------------------------------------------- QC: scale


def test_infer_scale_flags_raw_counts() -> None:
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.integers(0, 50_000, size=(50, 20)).astype(float))
    rep = infer_scale(counts)
    assert rep.scale == "linear"
    assert rep.assay == "rna-seq-counts"
    assert rep.recommendation is not None


def test_infer_scale_accepts_log_array_data() -> None:
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(8, 2, size=(50, 20)))
    rep = infer_scale(expr)
    assert rep.scale == "log"
    assert rep.recommendation is None


def test_infer_scale_detects_prestandardised_matrix() -> None:
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(0, 1, size=(50, 20)))
    rep = infer_scale(expr)
    assert rep.scale == "z"
    assert "cohort-relative" in (rep.recommendation or "")


def test_infer_scale_flags_linear_tpm() -> None:
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.gamma(2, 200, size=(50, 20)))
    rep = infer_scale(expr)
    assert rep.scale == "linear"
    assert "log2" in (rep.recommendation or "")


# ------------------------------------------------------------------- QC: cohort


def cohort_from(expr: pd.DataFrame, events: int = 20) -> Cohort:
    """Wrap a matrix in a cohort with a usable endpoint."""
    n = expr.shape[1]
    clinical = pd.DataFrame(
        {
            "time": np.linspace(1, 60, n),
            "event": [1] * events + [0] * (n - events),
        },
        index=list(expr.columns),
    )
    return Cohort(name="QC", expr=expr, clinical=clinical, provenance=PROV)


def test_clean_cohort_passes() -> None:
    rep = run_qc(cohort_from(matrix([f"G{i}" for i in range(60)])), event_col="event")
    assert rep.ok
    rep.raise_on_fail()


def test_small_cohort_fails() -> None:
    rep = run_qc(cohort_from(matrix([f"G{i}" for i in range(10)], n_samples=12), events=4))
    assert not rep.ok
    assert any(f.code == "n_samples" for f in rep.failures)
    with pytest.raises(QCFailure, match="QC failure"):
        rep.raise_on_fail()


def test_zero_variance_genes_warn() -> None:
    expr = matrix([f"G{i}" for i in range(40)])
    expr.iloc[0] = 5.0
    rep = run_qc(cohort_from(expr))
    assert any(f.code == "zero_variance" for f in rep.warnings)


def test_duplicate_gene_symbols_fail() -> None:
    expr = matrix([f"G{i}" for i in range(40)])
    expr.index = ["DUP"] * 2 + list(expr.index[2:])
    rep = run_qc(cohort_from(expr))
    assert any(f.code == "duplicate_genes" for f in rep.failures)


def test_sample_missingness_fails() -> None:
    expr = matrix([f"G{i}" for i in range(40)])
    expr.iloc[:, 0] = np.nan
    rep = run_qc(cohort_from(expr))
    assert any(f.code == "sample_missingness" for f in rep.failures)


def test_low_event_count_is_flagged() -> None:
    rep = run_qc(cohort_from(matrix([f"G{i}" for i in range(40)]), events=6), event_col="event")
    codes = [f.code for f in rep.findings]
    assert "event_count" in codes
    assert rep.summary["n_events"] == 6


def test_signature_coverage_is_reported_per_signature() -> None:
    sig = registry.load_bundled("smith20")
    present = list(sig.genes[:15]) + [f"G{i}" for i in range(30)]
    rep = run_qc(cohort_from(matrix(present)), signatures=[sig])
    finding = next(f for f in rep.findings if f.code == f"coverage:{sig.name}")
    assert finding.detail["coverage"] == pytest.approx(0.75)
    assert finding.level is Level.WARN
    assert finding.detail["checksum"] == sig.checksum


def test_full_coverage_is_info_not_warning() -> None:
    sig = registry.load_bundled("smith20")
    rep = run_qc(cohort_from(matrix(list(sig.genes))), signatures=[sig])
    finding = next(f for f in rep.findings if f.code.startswith("coverage:"))
    assert finding.level is Level.INFO


def test_report_renders_markdown_and_json() -> None:
    rep = run_qc(cohort_from(matrix([f"G{i}" for i in range(40)])), event_col="event")
    md = rep.to_markdown()
    assert "# QC report - QC" in md
    assert "population_hash" in md
    assert '"ok": true' in rep.to_json().lower()


def test_a_sampled_scale_range_is_labelled_as_sampled():
    """A subsampled range read as exact is a small but real trap.

    infer_scale subsamples large matrices, so its min/max are the sample's, not
    the matrix's. On a real cohort the difference was 15.06 against a true
    16.64 - harmless for classifying the scale, misleading if quoted.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from hypoxiapipe.qc.platform import infer_scale  # noqa: PLC0415

    small = pd.DataFrame(np.random.default_rng(0).normal(8, 1, (10, 10)))
    assert infer_scale(small).sampled is False
    assert infer_scale(small).to_dict()["range_is_sampled"] is False

    large = pd.DataFrame(np.random.default_rng(0).normal(8, 1, (600, 600)))
    report = infer_scale(large, sample_n=1000)
    assert report.sampled is True
    assert report.to_dict()["range_is_sampled"] is True
