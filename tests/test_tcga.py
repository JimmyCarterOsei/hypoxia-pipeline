"""TCGA ingest tests: GDC manifest, file assembly, CDR endpoints, full build.

All offline. The GDC responses are fixtures written into a tmp cache, which is
also how CI will run: hitting api.gdc.cancer.gov from a test suite makes the
suite fail whenever the API is slow, migrating, or unreachable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.errors import EndpointError, IngestError, ParseError
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.cdr import CDR_ENDPOINTS, check_endpoint, load_endpoint, read_cdr
from hypoxiapipe.ingest.gdc import (
    fetch_manifest,
    load_expression,
    manifest_query,
    parse_clinical,
    parse_manifest,
)
from hypoxiapipe.ingest.pipeline import build_cohort
from hypoxiapipe.ingest.spec import CohortSpec, load_bundled_cohort
from hypoxiapipe.ingest.tcga import (
    assemble_matrix,
    parse_barcode,
    parse_star_counts,
    restrict_to_primary_tumours,
    strip_ensembl_version,
    to_patient_level,
)
from hypoxiapipe.ingest.tcga_build import load_tcga

GENES = [
    ("ENSG00000149925.18", "ALDOA"),
    ("ENSG00000011426.10", "ANLN"),
    ("ENSG00000104313.15", "ESRP1"),
    ("ENSG00000155380.10", "SLC16A1"),
    ("ENSG00000142871.16", "CYR61"),
    ("ENSG00000170961.5", "STC2"),
    ("ENSG00000109046.14", "WSB1"),
    ("ENSG00000115548.14", "KDM3A"),
    ("ENSG00000072571.19", "HMMR"),
    ("ENSG00000134013.11", "LOXL2"),
]

#: Aliquot barcodes: eight primary tumours, one adjacent normal, and one
#: patient (CH-0002) contributing two aliquots.
ALIQUOTS = [
    "TCGA-CH-0001-01A-11R-1580-07",
    "TCGA-CH-0002-01A-11R-1580-07",
    "TCGA-CH-0002-01B-11R-1580-07",
    "TCGA-CH-0003-01A-11R-1580-07",
    "TCGA-CH-0004-01A-11R-1580-07",
    "TCGA-CH-0005-01A-11R-1580-07",
    "TCGA-CH-0006-01A-11R-1580-07",
    "TCGA-CH-0007-01A-11R-1580-07",
    "TCGA-CH-0008-01A-11R-1580-07",
    "TCGA-CH-0009-11A-11R-1580-07",  # solid tissue normal
]


def star_counts_file(seed: int) -> str:
    """Return a synthetic GDC STAR-counts file, including its N_ summary rows."""
    rng = np.random.default_rng(seed)
    lines = [
        "# gene-model: GENCODE v36",
        "gene_id\tgene_name\tgene_type\tunstranded\ttpm_unstranded\tfpkm_unstranded",
        "N_unmapped\t\t\t1000\t\t",
        "N_multimapping\t\t\t2000\t\t",
        "N_noFeature\t\t\t3000\t\t",
        "N_ambiguous\t\t\t4000\t\t",
    ]
    for gene_id, name in GENES:
        tpm = float(rng.lognormal(2.5, 0.8))
        lines.append(f"{gene_id}\t{name}\tprotein_coding\t{int(tpm * 30)}\t{tpm:.4f}\t{tpm:.4f}")
    return "\n".join(lines) + "\n"


def manifest_response() -> str:
    """Return a synthetic GDC /files response covering every aliquot."""
    hits = []
    for i, barcode in enumerate(ALIQUOTS):
        sample_type = "Solid Tissue Normal" if barcode[13:15] == "11" else "Primary Tumor"
        hits.append(
            {
                "file_id": f"uuid-{i:04d}",
                "file_name": f"{barcode}.rna_seq.augmented_star_gene_counts.tsv",
                "file_size": 4200000,
                "md5sum": f"{i:032x}",
                "analysis": {"workflow_type": "STAR - Counts"},
                "cases": [
                    {
                        "submitter_id": barcode[:12],
                        "samples": [
                            {
                                "sample_type": sample_type,
                                "portions": [
                                    {"analytes": [{"aliquots": [{"submitter_id": barcode}]}]}
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    return json.dumps({"data": {"hits": hits, "pagination": {"total": len(hits)}}})


def cdr_table() -> pd.DataFrame:
    """Return a synthetic TCGA-CDR table with PRAD and a decoy tumour type."""
    patients = sorted({b[:12] for b in ALIQUOTS})
    rng = np.random.default_rng(3)
    rows = []
    for i, patient in enumerate(patients):
        rows.append(
            {
                "bcr_patient_barcode": patient,
                "type": "PRAD",
                "age_at_initial_pathologic_diagnosis": 55 + i,
                "OS": 0,
                "OS.time": 900 + i * 40,
                "DSS": 0,
                "DSS.time": 900 + i * 40,
                "DFI": np.nan,
                "DFI.time": np.nan,
                "PFI": int(i % 2 == 0),
                "PFI.time": float(rng.integers(200, 2600)),
            }
        )
    # A different tumour type that must be filtered out by project.
    rows.append(
        {
            "bcr_patient_barcode": "TCGA-ZZ-9999",
            "type": "BRCA",
            "age_at_initial_pathologic_diagnosis": 60,
            "OS": 1,
            "OS.time": 500,
            "DSS": 1,
            "DSS.time": 500,
            "DFI": 1,
            "DFI.time": 400,
            "PFI": 1,
            "PFI.time": 400,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def gdc_cache(tmp_path):
    """Return an offline cache primed with a GDC manifest and its STAR files."""
    cache = Cache(tmp_path / "cache", offline=True)
    cache.put(
        "gdc/TCGA-PRAD/STAR_-_Counts_manifest.json",
        manifest_response().encode(),
        url="fixture://manifest",
    )
    for i in range(len(ALIQUOTS)):
        cache.put(f"gdc/files/uuid-{i:04d}", star_counts_file(i).encode(), url="fixture://star")
    return cache


@pytest.fixture
def cdr_path(tmp_path):
    """Write the synthetic CDR table to disk and return its path."""
    path = tmp_path / "TCGA-CDR.tsv"
    cdr_table().to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def prad_spec(cdr_path):
    """Return a TCGA-PRAD spec pointing at the fixture CDR table."""
    return CohortSpec(
        name="TCGA-PRAD",
        source="tcga",
        accession="TCGA-PRAD",
        clinical_source="cdr",
        clinical_path=str(cdr_path),
        cdr_endpoint="PFI",
        log2_transform=True,
    )


# --------------------------------------------------------------------------
# barcodes and STAR files
# --------------------------------------------------------------------------


def test_ensembl_versions_are_stripped():
    # Version suffixes change between GDC releases and silently break joins.
    assert strip_ensembl_version("ENSG00000141510.16") == "ENSG00000141510"
    assert strip_ensembl_version("ENSG00000141510") == "ENSG00000141510"


def test_barcode_parsing_separates_patient_from_sample_type():
    bc = parse_barcode("TCGA-CH-5761-01A-11R-1580-07")
    assert bc.patient == "TCGA-CH-5761"
    assert bc.is_primary_tumour

    assert not parse_barcode("TCGA-CH-5761-11A-11R-1580-07").is_primary_tumour
    with pytest.raises(ParseError, match="not a TCGA barcode"):
        parse_barcode("GSM1234567")


def test_star_summary_rows_are_not_genes():
    series = parse_star_counts(star_counts_file(0))
    assert len(series) == len(GENES)
    assert not any(str(i).startswith("N_") for i in series.index)
    assert "ALDOA" in series.index


def test_unknown_star_value_column_is_rejected():
    with pytest.raises(ValueError, match="unknown STAR value column"):
        parse_star_counts(star_counts_file(0), value_column="counts")


# --------------------------------------------------------------------------
# GDC manifest
# --------------------------------------------------------------------------


def test_manifest_query_requests_the_aliquot_barcode():
    query = manifest_query("TCGA-PRAD")
    # Without this field a file UUID cannot be tied to a patient at all.
    assert "aliquots.submitter_id" in query["fields"]
    assert query["filters"]["content"][0]["content"]["value"] == ["TCGA-PRAD"]


def test_parse_manifest_extracts_barcode_and_patient():
    manifest = parse_manifest(manifest_response())
    assert len(manifest) == len(ALIQUOTS)
    assert set(manifest.columns) >= {"file_id", "barcode", "patient", "sample_type"}
    assert manifest["patient"].nunique() == 9  # CH-0002 contributes two aliquots


def test_manifest_without_barcodes_fails_loudly():
    payload = json.loads(manifest_response())
    payload["data"]["hits"][0]["cases"][0]["samples"] = []
    with pytest.raises(ParseError, match="no aliquot barcode"):
        parse_manifest(json.dumps(payload))


def test_empty_manifest_is_an_error_not_an_empty_cohort():
    with pytest.raises(ParseError, match="no files"):
        parse_manifest(json.dumps({"data": {"hits": []}}))


def test_fetch_manifest_uses_the_cache_offline(gdc_cache):
    manifest, entry = fetch_manifest("TCGA-PRAD", gdc_cache)
    assert len(manifest) == len(ALIQUOTS)
    assert entry.checksum.startswith("sha256:")


def test_parse_clinical_is_patient_indexed():
    payload = json.dumps(
        {
            "data": {
                "hits": [
                    {
                        "submitter_id": "TCGA-CH-0001",
                        "demographic": {"vital_status": "Alive", "days_to_death": None},
                        "diagnoses": [{"days_to_last_follow_up": 900}],
                    }
                ]
            }
        }
    )
    clinical = parse_clinical(payload)
    assert clinical.index.tolist() == ["TCGA-CH-0001"]
    assert clinical.loc["TCGA-CH-0001", "days_to_last_follow_up"] == 900


# --------------------------------------------------------------------------
# assembly and sample selection
# --------------------------------------------------------------------------


def test_load_expression_refuses_partial_cohorts_by_default(gdc_cache):
    manifest, _ = fetch_manifest("TCGA-PRAD", gdc_cache)
    broken = manifest.copy()
    broken.loc[0, "file_id"] = "uuid-missing"

    with pytest.raises(IngestError, match="files failed"):
        load_expression(broken, gdc_cache)

    # Tolerating losses is possible, but only by asking for it.
    per_sample, report = load_expression(broken, gdc_cache, tolerate_failures=1)
    assert report.n_failed == 1
    assert len(per_sample) == len(ALIQUOTS) - 1


def test_normals_are_dropped_and_counted():
    expr = pd.DataFrame(
        np.arange(len(GENES) * len(ALIQUOTS), dtype=float).reshape(len(GENES), len(ALIQUOTS)),
        index=[g[1] for g in GENES],
        columns=ALIQUOTS,
    )
    kept, dropped = restrict_to_primary_tumours(expr)
    assert len(dropped) == 1
    assert dropped[0].startswith("TCGA-CH-0009")
    assert kept.shape[1] == len(ALIQUOTS) - 1


def test_multiple_aliquots_collapse_to_one_patient_column():
    expr = pd.DataFrame(
        [[1.0, 2.0, 4.0]],
        index=["ALDOA"],
        columns=ALIQUOTS[:3],
    )
    first, duplicated = to_patient_level(expr, rule="first")
    assert duplicated == ["TCGA-CH-0002"]
    assert list(first.columns) == ["TCGA-CH-0001", "TCGA-CH-0002"]
    assert first.loc["ALDOA", "TCGA-CH-0002"] == 2.0

    averaged, _ = to_patient_level(expr, rule="mean")
    assert averaged.loc["ALDOA", "TCGA-CH-0002"] == 3.0


def test_assemble_matrix_collapses_duplicate_symbols():
    a = pd.Series([1.0, 5.0], index=pd.Index(["ALDOA", "ALDOA"], name="gene"))
    b = pd.Series([2.0, 6.0], index=pd.Index(["ALDOA", "ALDOA"], name="gene"))
    frame = assemble_matrix({"S1": a, "S2": b})
    assert frame.shape[0] == 1


# --------------------------------------------------------------------------
# CDR endpoints
# --------------------------------------------------------------------------


def test_overall_survival_is_refused_for_prad():
    # ~12 deaths in 500 patients: a model fitted on this returns a hazard
    # ratio, a CI and a p-value that all look like results.
    with pytest.raises(EndpointError, match="not a usable endpoint"):
        check_endpoint("TCGA-PRAD", "OS")
    with pytest.raises(EndpointError, match="recommend PFI"):
        check_endpoint("TCGA-PRAD", "DSS")
    check_endpoint("TCGA-PRAD", "PFI")  # the recommended one passes


def test_unknown_endpoint_names_the_valid_ones():
    with pytest.raises(EndpointError, match="unknown CDR endpoint"):
        check_endpoint("TCGA-PRAD", "BCR")
    assert set(CDR_ENDPOINTS) == {"OS", "DSS", "DFI", "PFI"}


def test_cdr_loads_pfi_in_months_and_filters_by_project(cdr_path):
    clinical, report = load_endpoint(cdr_path, "TCGA-PRAD", endpoint="PFI")
    assert "TCGA-ZZ-9999" not in clinical.index  # the BRCA decoy is excluded
    assert report.n_project == 9
    assert report.n_events > 0
    # CDR times are days; the pipeline works in months.
    assert clinical["time_months"].max() < 90


def test_cdr_capping_censors_late_events(cdr_path):
    uncapped, uncapped_report = load_endpoint(
        cdr_path, "TCGA-PRAD", endpoint="PFI", cap_months=None
    )
    capped, capped_report = load_endpoint(cdr_path, "TCGA-PRAD", endpoint="PFI", cap_months=24.0)
    assert capped["time_months"].max() <= 24.0
    assert capped_report.n_events <= uncapped_report.n_events
    assert capped_report.n_censored_by_cap == uncapped_report.n_events - capped_report.n_events


def test_discouraged_endpoint_requires_a_deliberate_override(cdr_path):
    with pytest.raises(EndpointError):
        load_endpoint(cdr_path, "TCGA-PRAD", endpoint="OS")
    clinical, report = load_endpoint(cdr_path, "TCGA-PRAD", endpoint="OS", allow_discouraged=True)
    assert report.endpoint == "OS"


def test_missing_cdr_file_explains_where_to_get_it(tmp_path):
    with pytest.raises(IngestError, match="not redistributed"):
        read_cdr(tmp_path / "absent.xlsx")


def test_a_file_that_is_not_the_cdr_is_rejected(tmp_path):
    path = tmp_path / "wrong.tsv"
    pd.DataFrame({"patient": ["x"], "PFI": [1]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(IngestError, match="does not look like the TCGA-CDR"):
        read_cdr(path)


# --------------------------------------------------------------------------
# full assembly
# --------------------------------------------------------------------------


def test_load_tcga_assembles_a_patient_level_cohort(gdc_cache, prad_spec):
    cohort, report = load_tcga(prad_spec, gdc_cache, min_samples=5)

    # 10 aliquots -> 9 dropped normal -> 8 patients after deduplicating CH-0002
    assert report["manifest"]["n_files"] == 10
    assert report["selection"]["n_dropped_non_primary"] == 1
    assert report["selection"]["n_patients_with_multiple_aliquots"] == 1
    assert cohort.n_samples == 8
    assert all(len(str(c)) == 12 for c in cohort.expr.columns)
    assert "time_months" in cohort.clinical.columns


def test_tcga_loader_leaves_the_scale_to_the_pipeline(gdc_cache, prad_spec):
    """The loader must not log2; build_cohort's scale stage does it."""
    import numpy as np  # noqa: PLC0415

    cohort, _ = load_tcga(prad_spec, gdc_cache, min_samples=5)
    built = build_cohort(prad_spec, gdc_cache, min_samples=5).cohort
    # Compare the genes harmonisation left untouched (CYR61 becomes CCN1): the
    # loader's values must be the pipeline's pre-image under log2(x + 1), which
    # holds only if the loader left the scale alone.
    shared = [g for g in cohort.expr.index if g in built.expr.index]
    assert len(shared) >= 5
    np.testing.assert_allclose(
        np.log2(cohort.expr.loc[shared, built.expr.columns].to_numpy() + 1.0),
        built.expr.loc[shared].to_numpy(),
        rtol=1e-9,
    )


def test_expression_is_log2_transformed_exactly_once(gdc_cache, prad_spec):
    """Regression: the transform was applied by both the loader and the pipeline.

    Double-logging is silent - values stay positive and monotonic, so nothing
    errors and every downstream number is computed on log2(log2(TPM + 1) + 1).
    Caught only by the value range topping out near 4 instead of ~15 on real
    TCGA data.
    """
    import numpy as np  # noqa: PLC0415

    raw, _ = load_tcga(prad_spec, gdc_cache, min_samples=5)
    built = build_cohort(prad_spec, gdc_cache, min_samples=5).cohort

    expected_max = float(np.log2(raw.expr.to_numpy().max() + 1.0))
    actual_max = float(built.expr.to_numpy().max())
    assert actual_max == pytest.approx(expected_max, rel=0.02), (
        f"expected a single log2 (max ~{expected_max:.2f}), got {actual_max:.2f}"
    )
    # A second transform would compress the maximum below log2 of itself.
    assert actual_max > float(np.log2(expected_max + 1.0)) * 1.5


def test_aliquot_level_clinical_join_would_be_empty(gdc_cache, prad_spec):
    """The join only works because expression is truncated to patient level."""
    from dataclasses import replace  # noqa: PLC0415

    cohort, _ = load_tcga(prad_spec, gdc_cache, min_samples=5)
    aliquots = set(ALIQUOTS)
    patients = set(cohort.clinical.index)
    assert not (aliquots & patients), "aliquot barcodes must not match clinical directly"
    assert set(cohort.expr.columns) <= patients

    # And a spec that keeps normals gets more patients, not fewer.
    with_normals = replace(prad_spec, primary_tumours_only=False)
    wider, _ = load_tcga(with_normals, gdc_cache, min_samples=5)
    assert wider.n_samples >= cohort.n_samples


def test_build_cohort_runs_tcga_through_the_same_pipeline(gdc_cache, prad_spec):
    result = build_cohort(prad_spec, gdc_cache, min_samples=5)
    cohort = result.cohort

    assert result.tcga is not None
    assert result.tcga["endpoint"]["endpoint"] == "PFI"
    # Symbol harmonisation still runs: CYR61 is retired in favour of CCN1.
    assert result.symbols is not None
    assert "CCN1" in cohort.expr.index or "CYR61" in cohort.expr.index
    actions = [s.action for s in cohort.provenance.steps]
    assert "gdc_manifest" in actions
    assert "harmonise_symbols" in actions
    assert "analysis_set" in actions
    assert cohort.population_hash.startswith("sha256:")


def test_tcga_build_is_reproducible(gdc_cache, prad_spec):
    a = build_cohort(prad_spec, gdc_cache, min_samples=5)
    b = build_cohort(prad_spec, gdc_cache, min_samples=5)
    assert a.cohort.expr_checksum == b.cohort.expr_checksum
    assert a.cohort.population_hash == b.cohort.population_hash


def test_bundled_prad_spec_defaults_to_the_recommended_endpoint():
    spec = load_bundled_cohort("tcga-prad")
    assert spec.source == "tcga"
    assert spec.clinical_source == "cdr"
    assert spec.cdr_endpoint == "PFI"
    assert spec.tolerate_file_failures == 0


def test_cdr_source_without_a_path_is_rejected_at_spec_time():
    with pytest.raises(IngestError, match="requires clinical_path"):
        CohortSpec(name="X", source="tcga", accession="TCGA-PRAD", clinical_source="cdr")
    with pytest.raises(IngestError, match="clinical_source must be"):
        CohortSpec(name="X", source="tcga", accession="TCGA-PRAD", clinical_source="cbioportal")


def test_a_bad_clinical_path_fails_before_downloading_anything(gdc_cache, prad_spec, tmp_path):
    """Fail fast: the CDR is read last but validated first.

    Without the upfront check this surfaces after ~2 GB of expression files
    have been fetched, which is a bad way to learn that a path is wrong.
    """
    from dataclasses import replace  # noqa: PLC0415

    broken = replace(prad_spec, clinical_path=str(tmp_path / "absent.xlsx"))
    with pytest.raises(IngestError, match="not redistributed"):
        load_tcga(broken, gdc_cache, min_samples=5)


def test_a_discouraged_endpoint_fails_before_downloading_anything(gdc_cache, prad_spec):
    from dataclasses import replace  # noqa: PLC0415

    with pytest.raises(EndpointError, match="not a usable endpoint"):
        load_tcga(replace(prad_spec, cdr_endpoint="OS"), gdc_cache, min_samples=5)
