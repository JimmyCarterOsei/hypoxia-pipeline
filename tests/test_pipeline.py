"""Phase 2 tests: probe mapping, endpoints, cohort specs, store, end-to-end build.

Everything here runs offline against fixtures written into a tmp cache, because
a test suite that reaches NCBI is a test suite that fails on a train.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hypoxiapipe.errors import CacheMissError, EndpointError, IngestError, ParseError
from hypoxiapipe.harmonise.probes import (
    apply_probe_map,
    fetch_probe_map,
    gpl_annot_url,
    parse_gpl_annotation,
)
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.endpoints import EndpointSpec, derive_endpoint
from hypoxiapipe.ingest.pipeline import build_cohort
from hypoxiapipe.ingest.spec import CohortSpec, list_bundled_cohorts, load_bundled_cohort
from hypoxiapipe.ingest.store import load_cohort, save_cohort
from tests.conftest import GPL_ANNOT

# --------------------------------------------------------------------------
# probe mapping
# --------------------------------------------------------------------------


def test_gpl_url_derives_the_directory_stub():
    assert gpl_annot_url("GPL10558").endswith("GPL10nnn/GPL10558/annot/GPL10558.annot.gz")
    with pytest.raises(ValueError, match="not a GEO platform"):
        gpl_annot_url("GSE70768")


def test_parse_annotation_finds_symbol_column_and_classifies_probes():
    pm = parse_gpl_annotation(GPL_ANNOT)
    assert pm.platform == "GPL9999"
    assert pm.symbol_column == "Gene symbol"
    assert pm.mapping["PROBE_1"] == "ALDOA"
    # multi-target probes are held back, not silently resolved to their first gene
    assert "PROBE_4" not in pm.mapping
    assert pm.multi_target["PROBE_4"] == ("BNIP3", "BNIP3P1")
    # blank symbol is unmapped, not an empty-string gene
    assert "PROBE_5" in pm.unmapped


def test_multi_target_rule_first_is_available_but_not_default():
    pm = parse_gpl_annotation(GPL_ANNOT, multi="first")
    assert pm.mapping["PROBE_4"] == "BNIP3"
    with pytest.raises(ValueError, match="unknown multi-target rule"):
        parse_gpl_annotation(GPL_ANNOT, multi="guess")


def test_annotation_without_symbol_column_fails_loudly():
    with pytest.raises(ParseError, match="no gene-symbol column"):
        parse_gpl_annotation("ID\tDescription\nPROBE_1\tsomething\n")


def test_apply_probe_map_collapses_and_counts_what_it_dropped():
    pm = parse_gpl_annotation(GPL_ANNOT)
    expr = pd.DataFrame(
        # PROBE_2 has the higher mean, so max_mean should keep it for ALDOA
        [
            [1.0, 1.0],
            [9.0, 9.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ],
        index=[f"PROBE_{i}" for i in range(1, 9)],
        columns=["S1", "S2"],
    )
    mapped, report = apply_probe_map(expr, pm)
    assert list(mapped.loc["ALDOA"]) == [9.0, 9.0]
    assert report.n_dropped_multi == 1
    assert report.n_dropped_unmapped == 1
    assert report.collapsed == {"ALDOA": 2}
    assert mapped.shape[0] == 5  # ALDOA, CYR61, ANLN, ESRP1, SLC16A1


def test_probe_map_against_wrong_platform_fails_rather_than_returning_nothing():
    pm = parse_gpl_annotation(GPL_ANNOT)
    expr = pd.DataFrame([[1.0, 2.0]], index=["ILMN_1234567"], columns=["S1", "S2"])
    with pytest.raises(ParseError, match="no probes in the matrix matched"):
        apply_probe_map(expr, pm)


def test_fetch_probe_map_is_offline_when_cached(primed_cache):
    pm, url, checksum = fetch_probe_map("GPL9999", primed_cache)
    assert pm.n_mapped == 6
    assert checksum.startswith("sha256:")
    assert url.endswith("GPL9999.annot.gz")


def test_fetch_probe_map_offline_miss_raises(tmp_path):
    with pytest.raises(CacheMissError):
        fetch_probe_map("GPL1234", Cache(tmp_path, offline=True))


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


def test_time_unit_is_required_and_validated():
    with pytest.raises(EndpointError, match="unknown time_unit"):
        EndpointSpec(name="x", time_column="t", event_column="e", time_unit="fortnights")
    with pytest.raises(EndpointError, match="'time_unit' is required"):
        EndpointSpec.from_dict({"time_column": "t", "event_column": "e"})


def test_days_are_converted_to_months():
    clinical = pd.DataFrame({"t": ["365", "730"], "e": ["1", "0"]}, index=["A", "B"])
    spec = EndpointSpec(name="OS", time_column="t", event_column="e", time_unit="days")
    out, report = derive_endpoint(clinical, spec)
    assert out.loc["A", "time_months"] == pytest.approx(12.0, abs=0.05)
    assert report.n_usable == 2
    assert report.n_events == 1


def test_capping_censors_late_events_and_reports_the_count():
    clinical = pd.DataFrame({"t": ["70", "30", "80"], "e": ["1", "1", "0"]}, index=["A", "B", "C"])
    spec = EndpointSpec(
        name="BCR", time_column="t", event_column="e", time_unit="months", cap_months=60
    )
    out, report = derive_endpoint(clinical, spec)
    # A recurred at 70 months: beyond the horizon, so censored at 60
    assert out.loc["A", "event"] == 0.0
    assert out.loc["A", "time_months"] == 60.0
    assert out.loc["B", "event"] == 1.0
    assert report.n_censored_by_cap == 1
    assert report.n_events == 1


def test_unrecognised_event_values_are_reported_not_guessed():
    clinical = pd.DataFrame({"t": ["10", "20"], "e": ["relapsed", "0"]}, index=["A", "B"])
    spec = EndpointSpec(name="BCR", time_column="t", event_column="e", time_unit="months")
    out, report = derive_endpoint(clinical, spec)
    assert pd.isna(out.loc["A", "event"])
    assert report.n_unparsed_event == 1
    assert "relapsed" in report.unrecognised_values


def test_missing_endpoint_column_names_the_available_ones():
    clinical = pd.DataFrame({"followup": ["1"]}, index=["A"])
    spec = EndpointSpec(name="BCR", time_column="t", event_column="e", time_unit="months")
    with pytest.raises(EndpointError, match="not in clinical table"):
        derive_endpoint(clinical, spec)


# --------------------------------------------------------------------------
# cohort specs
# --------------------------------------------------------------------------


def test_bundled_cohort_specs_parse():
    bundled = list_bundled_cohorts()
    assert "cambridge" in bundled
    for name, item in bundled.items():
        assert isinstance(item, CohortSpec), f"{name}: {item}"


def test_cohort_spec_requires_accession_for_remote_sources():
    with pytest.raises(IngestError, match="requires an accession"):
        CohortSpec(name="X", source="geo")
    with pytest.raises(IngestError, match="requires a path"):
        CohortSpec(name="X", source="local")
    with pytest.raises(IngestError, match="unknown source"):
        CohortSpec(name="X", source="dropbox", accession="Y")


def test_cambridge_spec_pins_platform_and_five_year_cap():
    spec = load_bundled_cohort("cambridge")
    assert spec.platform == "GPL10558"
    assert spec.endpoint is not None
    assert spec.endpoint.cap_months == 60


# --------------------------------------------------------------------------
# end-to-end build
# --------------------------------------------------------------------------


def test_build_cohort_end_to_end(primed_cache, geo_spec):
    result = build_cohort(geo_spec, primed_cache, min_samples=10)
    cohort = result.cohort

    # probes became symbols, and the pinned authority renamed the retired one
    assert "ALDOA" in cohort.expr.index
    assert "CCN1" in cohort.expr.index or "CYR61" in cohort.expr.index
    assert not any(str(g).startswith("PROBE_") for g in cohort.expr.index)

    # endpoint derived and the cohort restricted to samples that have it
    assert "time_months" in cohort.clinical.columns
    assert result.endpoint is not None
    assert result.endpoint.n_events > 0
    assert cohort.n_samples == result.endpoint.n_usable

    # every stage left a trace
    actions = [s.action for s in cohort.provenance.steps]
    assert actions.index("map_probes_to_symbols") < actions.index("harmonise_symbols")
    assert "analysis_set" in actions
    assert cohort.provenance.symbol_authority


def test_build_records_the_population_scores_will_be_relative_to(primed_cache, geo_spec):
    result = build_cohort(geo_spec, primed_cache, min_samples=10)
    assert result.cohort.population_hash.startswith("sha256:")
    assert result.qc.summary["population_hash"] == result.cohort.population_hash


def test_build_fails_when_the_cohort_contradicts_its_pinned_expectations(primed_cache, geo_spec):
    from dataclasses import replace

    from hypoxiapipe.ingest.spec import Expectation

    strict = replace(geo_spec, expect=Expectation(n_samples=999))
    with pytest.raises(IngestError, match="does not match its pinned expectations"):
        build_cohort(strict, primed_cache, min_samples=10)

    lenient = build_cohort(strict, primed_cache, min_samples=10, strict_expectations=False)
    assert lenient.expectation_failures


def test_build_is_reproducible(primed_cache, geo_spec):
    a = build_cohort(geo_spec, primed_cache, min_samples=10)
    b = build_cohort(geo_spec, primed_cache, min_samples=10)
    assert a.cohort.expr_checksum == b.cohort.expr_checksum
    assert a.cohort.population_hash == b.cohort.population_hash


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


def test_cohort_round_trips_through_disk(primed_cache, geo_spec, tmp_path):
    built = build_cohort(geo_spec, primed_cache, min_samples=10).cohort
    save_cohort(built, tmp_path / "cohort")
    loaded = load_cohort(tmp_path / "cohort")

    assert loaded.expr_checksum == built.expr_checksum
    assert loaded.population_hash == built.population_hash
    assert loaded.n_samples == built.n_samples
    assert [s.action for s in loaded.provenance.steps] == [s.action for s in built.provenance.steps]
    meta = json.loads((tmp_path / "cohort" / "cohort.json").read_text())
    assert meta["provenance"]["symbol_authority"]


def test_edited_matrix_on_disk_fails_to_load(primed_cache, geo_spec, tmp_path):
    built = build_cohort(geo_spec, primed_cache, min_samples=10).cohort
    save_cohort(built, tmp_path / "cohort")

    tampered = pd.read_parquet(tmp_path / "cohort" / "expression.parquet")
    tampered.iloc[0, 0] = tampered.iloc[0, 0] + 1.0
    tampered.to_parquet(tmp_path / "cohort" / "expression.parquet")

    with pytest.raises(IngestError, match="checksum mismatch"):
        load_cohort(tmp_path / "cohort")
    # the escape hatch exists, but you have to ask for it
    assert load_cohort(tmp_path / "cohort", verify=False).n_samples == built.n_samples


def test_loading_a_non_cohort_directory_says_what_is_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestError, match="missing expression.parquet"):
        load_cohort(tmp_path / "empty")
