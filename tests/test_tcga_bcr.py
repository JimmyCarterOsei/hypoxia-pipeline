"""Offline longitudinal BCR regressions; all clinical examples are synthetic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.errors import IngestError
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.spec import CohortSpec, load_bundled_cohort
from hypoxiapipe.ingest.store import load_cohort, save_cohort
from hypoxiapipe.ingest.tcga_bcr import (
    ClinicalRecord,
    derive_endpoint,
    load_endpoint,
    read_records,
)

FIRST = "days_to_first_biochemical_recurrence"
SECOND = "days_to_second_biochemical_recurrence"
CONTACT = "days_to_last_followup"
NEW_EVENT = "new_tumor_event_after_initial_treatment"
TYPE = "new_neoplasm_event_type"
STATUS = "biochemical_recurrence"


def record(record_id="enrollment", patient="TCGA-AA-0001", **fields):
    return ClinicalRecord(
        patient=patient,
        record=record_id,
        kind="enrollment" if record_id == "enrollment" else "follow_up",
        fields={k: (str(v) if v is not None else None,) for k, v in fields.items()},
        procurement_status={
            k: ("Not Available" if v is None else "Completed",) for k, v in fields.items()
        },
        source=patient + ".xml",
        checksum="sha256:" + hashlib.sha256(patient.encode()).hexdigest(),
    )


def result(*records):
    clinical, report = derive_endpoint(records)
    return clinical, report.to_dict()


def test_enrollment_event_and_3044_conversion():
    c, r = result(record(**{STATUS: "YES", FIRST: 304.4, CONTACT: 1900}))
    assert c.iloc[0].time_months == pytest.approx(10)
    assert c.iloc[0].event == 1
    assert r["n_explicit_bcr"] == r["n_timed_bcr"] == 1
    assert r["n_after_expression_match"] is None


@pytest.mark.parametrize("initial_status", [None, "NO"])
def test_recovered_blank_or_historical_no_with_followup_bcr(initial_status):
    c, r = result(
        record(**{STATUS: initial_status, CONTACT: 500}),
        record("F1", **{TYPE: "Biochemical evidence of disease", FIRST: 380}),
    )
    assert c.iloc[0].event == 1
    assert c.iloc[0].time_days == 380
    assert c.iloc[0].selected_record == "F1"
    if initial_status == "NO":
        assert r["patients"][0]["conflicts"][0]["records"] == ["F1", "enrollment"]


def test_multiple_first_dates_choose_earliest_regardless_of_record_order():
    observations = [
        record(**{STATUS: "NO", CONTACT: 2500}),
        record("F1", **{TYPE: "Biochemical evidence of disease", FIRST: 2162}),
        record("F2", **{FIRST: 1155}),
    ]
    c, r = result(*observations)
    reverse, rr = result(*reversed(observations))
    pd.testing.assert_frame_equal(c, reverse)
    assert r == rr
    assert c.iloc[0].time_days == 1155
    assert any(x["kind"] == "differing_first_bcr_days" for x in r["patients"][0]["conflicts"])


@pytest.mark.parametrize("days,event", [(60 * 30.44, 1), (60 * 30.44 + 0.01, 0), (2310, 0)])
def test_administrative_cap_is_strictly_after_60_months(days, event):
    c, r = result(record(**{STATUS: "YES", FIRST: days}))
    assert c.iloc[0].event == event
    assert c.iloc[0].event_uncapped == 1
    assert c.iloc[0].time_months == pytest.approx(60)
    assert r["n_censored_by_cap"] == 1 - event


def test_explicit_followup_negative_and_later_unknown_do_not_extend_time():
    c, r = result(
        record(**{STATUS: None, CONTACT: 100}),
        record("F1", **{NEW_EVENT: "NO", CONTACT: 304.4}),
        record("F2", **{NEW_EVENT: None, CONTACT: 2000}),
    )
    assert c.iloc[0].time_months == pytest.approx(10)
    assert c.iloc[0].selected_record == "F1"
    assert r["n_supported_censored"] == 1


def test_enrollment_negative_keeps_its_own_date_not_later_unknown():
    c, _ = result(
        record(**{NEW_EVENT: "NO", CONTACT: 35}),
        record("F1", **{NEW_EVENT: None, CONTACT: 473}),
    )
    assert c.iloc[0].time_days == 35


def test_latest_of_multiple_explicit_negative_dates_is_used_and_capped():
    c, r = result(
        record(**{STATUS: "NO", CONTACT: 200}),
        record("F1", **{NEW_EVENT: "NO", CONTACT: 2400}),
        record("F2", **{STATUS: "NO", CONTACT: 500}),
    )
    assert c.iloc[0].time_days == 2400
    assert c.iloc[0].time_months == 60
    assert r["n_followup_capped"] == 1
    assert r["n_censored_by_cap"] == 0


@pytest.mark.parametrize(
    "fields",
    [
        {STATUS: ""},
        {STATUS: None},
        {"person_neoplasm_cancer_status": "TUMOR FREE"},
        {"primary_therapy_outcome_success": "Complete Remission/Response"},
        {"has_follow_ups_information": "YES"},
        {"has_new_tumor_events_information": "NO"},
        {"radiation_therapy": "NO"},
        {TYPE: "Locoregional Recurrence"},
        {TYPE: "Distant Metastasis"},
        {TYPE: "New Primary Tumor"},
        {FIRST: None},
    ],
)
def test_no_inferred_negative_from_non_endpoint_information(fields):
    c, r = result(record(**{**fields, CONTACT: 1000}))
    assert c.empty
    assert r["n_unknown_excluded"] == 1
    assert r["n_explicit_bcr"] == 0


def test_locoregional_event_does_not_erase_earlier_negative_or_become_bcr():
    c, _ = result(
        record(**{NEW_EVENT: "NO", CONTACT: 1712}),
        record("F1", **{TYPE: "Locoregional Recurrence", CONTACT: 2535}),
    )
    assert c.iloc[0].event == 0
    assert c.iloc[0].time_days == 1712


@pytest.mark.parametrize(
    "positive",
    [
        {STATUS: "YES"},
        {TYPE: "Biochemical evidence of disease"},
        {SECOND: 929},
        {"days_to_third_biochemical_recurrence": 1000},
        {"type_ofprogression_of_disease_ht": "Biochemical Recurrence"},
    ],
)
def test_explicit_untimed_bcr_never_falls_back_to_negative(positive):
    c, r = result(record(**{STATUS: "NO", CONTACT: 100}), record("F1", **positive))
    assert c.empty
    assert r["n_explicit_bcr"] == r["n_untimed_bcr_excluded"] == 1
    assert r["n_supported_censored"] == 0
    assert r["patients"][0]["reason"] == "explicit_bcr_without_valid_first_bcr_day"


@pytest.mark.parametrize("time", [0, -1, float("nan"), float("inf"), "bad", None])
def test_unusable_time_is_excluded_for_event_and_negative(time):
    c, r = result(record(**{STATUS: "YES", FIRST: time}))
    assert c.empty and r["n_untimed_bcr_excluded"] == 1
    c, r = result(record(**{STATUS: "NO", CONTACT: time}))
    assert c.empty and r["n_negative_without_time"] == 1


def test_populated_invalid_bcr_day_cannot_become_censored():
    c, r = result(record(**{STATUS: "NO", FIRST: "invalid", CONTACT: 200}))
    assert c.empty and r["n_untimed_bcr_excluded"] == 1


def write_xml(path, patient="TCGA-AA-0001", enrollment=None, followups=()):
    fields = {STATUS: None, FIRST: None, CONTACT: None, **(enrollment or {})}

    def elements(mapping):
        return "".join(
            f'<p:{k} procurement_status="{("Not Available" if v is None else "Completed")}">'
            f"{escape(str(v)) if v is not None else ''}</p:{k}>"
            for k, v in mapping.items()
        )

    fu = "".join(
        "<f:follow_up>"
        + elements({"bcr_followup_barcode": f"{patient}-F{i}", **mapping})
        + "</f:follow_up>"
        for i, mapping in enumerate(followups)
    )
    path.write_text(
        '<root xmlns:p="urn:patient" xmlns:f="urn:followup"><p:patient>'
        f"<p:bcr_patient_barcode>{patient}</p:bcr_patient_barcode>"
        + elements(fields)
        + "<p:follow_ups>"
        + fu
        + "</p:follow_ups></p:patient></root>"
    )
    return path


def test_xml_record_boundaries_checksums_and_provenance(tmp_path):
    path = write_xml(
        tmp_path / "clinical.xml",
        enrollment={CONTACT: 35, NEW_EVENT: "NO"},
        followups=[{CONTACT: 473, NEW_EVENT: None}],
    )
    records = read_records(path)
    assert len(records) == 2
    assert records[0].values(CONTACT) == ("35",)
    assert records[0].procurement_status[STATUS] == ("Not Available",)
    assert records[0].checksum == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    c, r = load_endpoint(path)
    assert c.iloc[0].time_days == 35
    ledger = r.to_dict()["patients"][0]
    assert ledger["selected_record"] == "enrollment"
    assert ledger["records"][1]["fields"][CONTACT] == ("473",)
    assert len(r.sources) == 1
    json.dumps(r.to_dict(), allow_nan=False)
    _, same = load_endpoint(path, expected_checksum=r.checksum)
    assert same.checksum == r.checksum
    with pytest.raises(IngestError, match="checksum mismatch"):
        load_endpoint(path, expected_checksum="sha256:wrong")


def test_missing_invalid_or_wrong_clinical_source_fails_loudly(tmp_path):
    with pytest.raises(IngestError, match="clinical XML"):
        read_records(tmp_path)
    path = tmp_path / "invalid.xml"
    path.write_text("not xml")
    with pytest.raises(IngestError, match="invalid BCR XML"):
        read_records(path)
    path.write_text(
        "<root><patient><bcr_patient_barcode>p1</bcr_patient_barcode><PFI>1</PFI></patient></root>"
    )
    with pytest.raises(IngestError, match="missing required fields"):
        read_records(path)
    path.write_text("<!DOCTYPE root><root/>")
    with pytest.raises(IngestError, match="DTD/entity"):
        read_records(path)


def test_duplicate_patients_and_observations_fail(tmp_path):
    write_xml(tmp_path / "a.xml")
    write_xml(tmp_path / "b.xml")
    with pytest.raises(IngestError, match="duplicate BCR patient"):
        read_records(tmp_path)
    with pytest.raises(IngestError, match="duplicate BCR observation"):
        result(record(), record())
    with pytest.raises(IngestError, match="no BCR clinical records"):
        result()


def test_clinical_and_matched_counts_are_distinct():
    records = [
        record(patient="p1", **{FIRST: 380}),
        record(patient="p2", **{FIRST: 2310}),
        record(patient="p3", **{STATUS: "YES"}),
        record(patient="p4", **{STATUS: None, CONTACT: 200}),
        record(patient="p5", **{STATUS: "NO", CONTACT: 200}),
    ]
    c, r = derive_endpoint(records)
    r.record_expression_match(c, ["p1", "p5"])
    d = r.to_dict()
    assert {
        k: d[k]
        for k in [
            "n_source",
            "n_explicit_bcr",
            "n_timed_bcr",
            "n_untimed_bcr_excluded",
            "n_supported_censored",
            "n_unknown_excluded",
            "n_before_expression_match",
            "n_after_expression_match",
            "n_events_before_cap",
            "n_events_after_cap",
            "n_censored_by_cap",
        ]
    } == dict(
        zip(
            [
                "n_source",
                "n_explicit_bcr",
                "n_timed_bcr",
                "n_untimed_bcr_excluded",
                "n_supported_censored",
                "n_unknown_excluded",
                "n_before_expression_match",
                "n_after_expression_match",
                "n_events_before_cap",
                "n_events_after_cap",
                "n_censored_by_cap",
            ],
            [5, 3, 2, 1, 1, 1, 3, 2, 2, 1, 1],
            strict=True,
        )
    )
    assert d["n_matched_events_before_cap"] == d["n_matched_events_after_cap"] == 1
    assert len(d["patients"]) == 5


def test_tcga_bcr_build_aligns_before_scoring_and_roundtrips_provenance(tmp_path):
    # Reuse the existing cached STAR fixture format without any live fetch.
    from hypoxiapipe.ingest.pipeline import build_cohort
    from tests.test_tcga import ALIQUOTS, manifest_response, star_counts_file

    cache = Cache(tmp_path / "cache", offline=True)
    cache.put("gdc/TCGA-PRAD/STAR_-_Counts_manifest.json", manifest_response().encode())
    for i in range(len(ALIQUOTS)):
        cache.put(f"gdc/files/uuid-{i:04d}", star_counts_file(i).encode())
    xml = tmp_path / "xml"
    xml.mkdir()
    for i in range(1, 5):
        fields = {FIRST: 380} if i == 1 else {STATUS: "NO", CONTACT: 800}
        if i == 4:
            fields = {STATUS: None, CONTACT: 900}
        write_xml(xml / f"{i}.xml", patient=f"TCGA-CH-{i:04d}", enrollment=fields)
    spec = CohortSpec(
        name="BCR",
        source="tcga",
        accession="TCGA-PRAD",
        clinical_source="tcga_bcr",
        clinical_path=str(xml),
        log2_transform=False,
    )
    built = build_cohort(spec, cache, min_samples=2)
    cohort = built.cohort
    assert list(cohort.expr.columns) == [f"TCGA-CH-{i:04d}" for i in range(1, 4)]
    # Actual cohort scoring is performed only over the three eligible patients.
    from hypoxiapipe.scoring import score
    from hypoxiapipe.signatures.registry import Signature

    sig = Signature(name="test", genes=("ALDOA", "ANLN"), scoring="rowmean")
    scores = score(cohort.expr, sig, method="rowmean", min_genes=2)
    assert len(scores.scores) == 3
    expected = cohort.expr.loc[list(sig.genes)].T
    expected = ((expected - expected.mean()) / expected.std(ddof=1)).mean(axis=1)
    np.testing.assert_allclose(scores.scores, expected)
    save_cohort(cohort, tmp_path / "built", extra={"build": built.to_dict()})
    loaded = load_cohort(tmp_path / "built")
    detail = next(s.detail for s in loaded.provenance.steps if s.action == "clinical")
    assert detail["n_after_expression_match"] == 3
    assert len(detail["patients"]) == 4  # excluded clinical records survive persistence


def test_bcr_preflight_fails_before_expression_fetch(tmp_path):
    from hypoxiapipe.ingest.tcga_build import load_tcga

    spec = CohortSpec(
        name="BCR",
        source="tcga",
        accession="TCGA-PRAD",
        clinical_source="tcga_bcr",
        clinical_path=str(tmp_path / "absent"),
    )
    with pytest.raises(IngestError, match="clinical XML"):
        load_tcga(spec, Cache(tmp_path / "cache", offline=True))


def test_existing_pfi_spec_is_byte_for_byte_unchanged():
    path = Path(__file__).parents[1] / "src/hypoxiapipe/ingest/specs/tcga-prad.yaml"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "dd03e6f150f9c1eb65d5cfc84b324f2d287e54f773c1b8e45195ebd11d1c67f3"
    )
    assert load_bundled_cohort("tcga-prad").clinical_source == "cdr"


def test_bcr_spec_is_distinct_and_not_pinned_to_historical_expression():
    spec = load_bundled_cohort("tcga-prad-bcr")
    assert spec.clinical_source == "tcga_bcr"
    assert spec.expect.n_samples == 483  # pinned only after the full STAR build
    assert spec.name != load_bundled_cohort("tcga-prad").name
    with pytest.raises(IngestError, match="requires TCGA-PRAD"):
        replace(spec, accession="TCGA-BRCA")
    with pytest.raises(IngestError, match="fixed 60-month cap"):
        replace(spec, endpoint=replace(spec.endpoint, cap_months=48))


@pytest.fixture(scope="module")
def audited_clinical_snapshot():
    import os

    path = os.environ.get("HYPOXIAPIPE_BCR_XML")
    if not path:
        pytest.skip("set HYPOXIAPIPE_BCR_XML to run the audited clinical snapshot regression")
    ref = json.loads((Path(__file__).parent / "fixtures/tcga_bcr_reference.json").read_text())
    clinical, report = load_endpoint(path, expected_checksum=ref["clinical_checksum"])
    return clinical, report, ref


def test_audited_clinical_snapshot_counts(audited_clinical_snapshot):
    _, report, ref = audited_clinical_snapshot
    assert {k: report.counts[k] for k in ref["clinical"]} == ref["clinical"]


def test_exact_audited_saved_expression_reproduces_431_92(audited_clinical_snapshot):
    import os
    import shutil
    import subprocess

    from hypoxiapipe.ingest.cache import sha256_file
    from hypoxiapipe.signatures.registry import load_bundled

    clinical, report, ref = audited_clinical_snapshot
    path = os.environ.get("HYPOXIAPIPE_BCR_EXPRESSION_RDS")
    rscript = shutil.which("Rscript")
    if not path or not rscript:
        pytest.skip(
            "set HYPOXIAPIPE_BCR_EXPRESSION_RDS and provide Rscript for snapshot regression"
        )
    assert sha256_file(Path(path)) == ref["historical_expression_rds_sha256"]
    code = (
        "x <- readRDS(commandArgs(TRUE)[1]); "
        'cat(paste(colnames(x), collapse="\\n")); cat("\\n--GENES--\\n"); '
        'cat(paste(rownames(x), collapse="\\n"))'
    )
    output = subprocess.run(
        [rscript, "-e", code, path], check=True, capture_output=True, text=True
    ).stdout
    patients, genes = output.split("\n--GENES--\n")
    ids = patients.splitlines()
    assert len(ids) == len(set(ids)) == 433
    shared = [p for p in ids if p in clinical.index]
    report.record_expression_match(clinical, shared)
    assert {k: report.counts[k] for k in ref["expression_matched"]} == ref["expression_matched"]
    assert set(load_bundled("smith20").genes) <= set(genes.splitlines())
