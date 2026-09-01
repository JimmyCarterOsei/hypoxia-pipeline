"""Ingest: cache behaviour, GEO series-matrix parsing, TCGA assembly.

Every test here runs offline against committed fixtures. Nothing in the suite
touches the network, so CI cannot go red because NCBI is slow.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from hypoxiapipe.errors import CacheMissError, ParseError
from hypoxiapipe.ingest import geo, tcga
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.cohort import Cohort, Provenance

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def geo_text() -> str:
    """Decompressed synthetic series-matrix text."""
    return gzip.decompress((FIXTURES / "GSE999999_series_matrix.txt.gz").read_bytes()).decode()


# --------------------------------------------------------------------------- cache


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    entry = cache.put("geo/x.txt", b"hello", url="https://example.org/x")
    assert cache.has("geo/x.txt")
    assert cache.get("geo/x.txt").checksum == entry.checksum
    assert entry.checksum.startswith("sha256:")


def test_cache_records_metadata_sidecar(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("geo/x.txt", b"hello", url="https://example.org/x")
    assert cache.meta_for("geo/x.txt").is_file()
    assert cache.get("geo/x.txt").url == "https://example.org/x"


def test_offline_cache_miss_raises_rather_than_downloading(tmp_path: Path) -> None:
    cache = Cache(tmp_path, offline=True)
    with pytest.raises(CacheMissError, match="offline mode"):
        cache.fetch("geo/missing.txt", "https://example.org/missing.txt")


def test_cache_hit_does_not_call_downloader(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("k", b"data")
    calls: list[str] = []

    def boom(url: str) -> bytes:
        calls.append(url)
        raise AssertionError("downloader must not run on a cache hit")

    assert cache.fetch("k", "https://example.org", downloader=boom).path.read_bytes() == b"data"
    assert calls == []


def test_cache_fetch_downloads_once_then_serves_from_disk(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    calls: list[str] = []

    def fake(url: str) -> bytes:
        calls.append(url)
        return b"payload"

    cache.fetch("k", "https://example.org/f", downloader=fake)
    cache.fetch("k", "https://example.org/f", downloader=fake)
    assert len(calls) == 1


# ----------------------------------------------------------------------------- GEO


def test_geo_url_construction() -> None:
    url = geo.series_matrix_url("GSE70768")
    assert "GSE70nnn/GSE70768/matrix/GSE70768_series_matrix.txt.gz" in url


def test_geo_url_rejects_non_series_accession() -> None:
    with pytest.raises(ValueError, match="not a GEO series accession"):
        geo.series_matrix_url("GSM12345")


def test_parse_series_matrix_shape_and_ids(geo_text: str) -> None:
    parsed = geo.parse_series_matrix(geo_text)
    assert parsed.expr.shape == (6, 36)
    assert parsed.platform == "GPL10558"
    assert all(str(c).startswith("GSM") for c in parsed.expr.columns)


def test_parse_series_matrix_extracts_characteristics(geo_text: str) -> None:
    """Characteristics are key: value pairs, parsed by key rather than position."""
    parsed = geo.parse_series_matrix(geo_text)
    assert {"bcr_time_(months)", "bcr_event", "gleason"} <= set(parsed.clinical.columns)
    assert parsed.clinical["gleason"].notna().all()


def test_parse_series_matrix_values_are_numeric(geo_text: str) -> None:
    parsed = geo.parse_series_matrix(geo_text)
    assert parsed.expr.dtypes.map(pd.api.types.is_numeric_dtype).all()


def test_parse_series_matrix_without_table_raises() -> None:
    with pytest.raises(ParseError, match="no expression table"):
        geo.parse_series_matrix('!Series_title\t"nothing here"')


def test_geo_cohort_from_cached_fixture(tmp_path: Path) -> None:
    """load_geo works end to end offline when the fixture is seeded into the cache."""
    cache = Cache(tmp_path, offline=True)
    cache.put(
        "geo/GSE999999_series_matrix.txt.gz",
        (FIXTURES / "GSE999999_series_matrix.txt.gz").read_bytes(),
        url="fixture",
    )
    cohort = geo.load_geo("GSE999999", cache, name="Fixture")
    assert isinstance(cohort, Cohort)
    assert cohort.n_samples == 36
    assert cohort.provenance.platform == "GPL10558"
    assert cohort.provenance.steps[0].action == "download"


# ---------------------------------------------------------------------------- TCGA


def test_strip_ensembl_version() -> None:
    assert tcga.strip_ensembl_version("ENSG00000141510.16") == "ENSG00000141510"


def test_parse_barcode_patient_and_type() -> None:
    bc = tcga.parse_barcode("TCGA-CH-5761-01A-11R-1580-07")
    assert bc.patient == "TCGA-CH-5761"
    assert bc.sample_type == "01"
    assert bc.is_primary_tumour


def test_parse_barcode_rejects_nonsense() -> None:
    with pytest.raises(ParseError, match="not a TCGA barcode"):
        tcga.parse_barcode("SAMPLE_1")


def test_parse_star_counts_drops_summary_rows() -> None:
    series = tcga.parse_star_counts((FIXTURES / "star_counts_sample.tsv").read_text())
    assert len(series) == 5
    assert not any(str(i).startswith("N_") for i in series.index)


def test_parse_star_counts_rejects_unknown_column() -> None:
    with pytest.raises(ValueError, match="unknown STAR value column"):
        tcga.parse_star_counts("x", value_column="not_a_column")


def test_assemble_matrix_collapses_duplicate_symbols() -> None:
    """Two Ensembl IDs mapping to ALDOA must not produce two ALDOA rows."""
    text = (FIXTURES / "star_counts_sample.tsv").read_text()
    per_sample = {
        f"TCGA-AA-000{i}-01A": tcga.parse_star_counts(text) * (1 + i / 10) for i in range(3)
    }
    matrix = tcga.assemble_matrix(per_sample)
    assert matrix.index.is_unique
    assert "ALDOA" in matrix.index
    # max_mean keeps the higher-expressed of the two ALDOA rows
    assert matrix.loc["ALDOA"].iloc[0] == pytest.approx(120.5)


def test_restrict_to_primary_tumours_drops_normals() -> None:
    expr = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=["G1"],
        columns=["TCGA-AA-0001-01A", "TCGA-AA-0002-11A", "TCGA-AA-0003-01A"],
    )
    kept, dropped = tcga.restrict_to_primary_tumours(expr)
    assert list(kept.columns) == ["TCGA-AA-0001-01A", "TCGA-AA-0003-01A"]
    assert dropped == ["TCGA-AA-0002-11A"]


def test_to_patient_level_reports_duplicate_aliquots() -> None:
    expr = pd.DataFrame(
        [[1.0, 3.0, 5.0]],
        index=["G1"],
        columns=["TCGA-AA-0001-01A", "TCGA-AA-0001-01B", "TCGA-AA-0002-01A"],
    )
    out, dupes = tcga.to_patient_level(expr, rule="mean")
    assert dupes == ["TCGA-AA-0001"]
    assert out.loc["G1", "TCGA-AA-0001"] == pytest.approx(2.0)
    assert out.shape[1] == 2


def test_log2_transform_matches_definition() -> None:
    expr = pd.DataFrame([[0.0, 1.0, 3.0]], index=["G1"], columns=["a", "b", "c"])
    out = tcga.log2_transform(expr)
    assert out.loc["G1"].tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_tcga_cohort_alignment_needs_barcode_truncation() -> None:
    """Aliquot barcodes will not join to patient-level clinical data."""
    expr = pd.DataFrame(
        [[1.0] * 40],
        index=["G1"],
        columns=[f"TCGA-AA-{i:04d}-01A-11R-1580-07" for i in range(40)],
    )
    clinical = pd.DataFrame(
        {"time": [1.0] * 40, "event": [0] * 40},
        index=[f"TCGA-AA-{i:04d}" for i in range(40)],
    )
    with pytest.raises(Exception, match="samples align"):
        Cohort.align("TCGA", expr, clinical, Provenance(source="GDC"))

    patient_level, _ = tcga.to_patient_level(expr)
    cohort = Cohort.align("TCGA", patient_level, clinical, Provenance(source="GDC"))
    assert cohort.n_samples == 40
