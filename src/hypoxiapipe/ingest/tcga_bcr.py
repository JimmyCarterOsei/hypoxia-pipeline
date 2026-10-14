"""Longitudinal TCGA-PRAD biochemical recurrence from clinical supplement XML.

Enrollment and follow-up observations retain their own contact dates. This is
neither CDR PFI nor the historical patient-table ``blank == NO`` shortcut.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from hypoxiapipe.errors import IngestError

DAYS_PER_MONTH = 30.44
CAP_MONTHS = 60.0
DAY_FIELDS = (
    "days_to_first_biochemical_recurrence",
    "days_to_second_biochemical_recurrence",
    "days_to_third_biochemical_recurrence",
)
TYPE_FIELDS = ("new_neoplasm_event_type", "type_ofprogression_of_disease_ht")
NEGATIVE_FIELDS = ("biochemical_recurrence", "new_tumor_event_after_initial_treatment")
CONTACT_FIELD = "days_to_last_followup"
FIELD_MAPPING = {
    "patient": "bcr_patient_barcode",
    "event_status": "biochemical_recurrence",
    "event_types": list(TYPE_FIELDS),
    "event_day_evidence": list(DAY_FIELDS),
    "first_event_day": DAY_FIELDS[0],
    "negative_status": list(NEGATIVE_FIELDS),
    "negative_contact_day": CONTACT_FIELD,
}


@dataclass(frozen=True)
class ClinicalRecord:
    """One original observation; repeated XML values and missingness are preserved."""

    patient: str
    record: str
    kind: str
    fields: dict[str, tuple[str | None, ...]]
    source: str
    checksum: str
    procurement_status: dict[str, tuple[str | None, ...]]

    def values(self, key: str) -> tuple[str, ...]:
        """Return populated values without turning missing answers into negatives."""
        return tuple(v.strip() for v in self.fields.get(key, ()) if v and v.strip())


def _checksum(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _leaves(element: ET.Element) -> Iterable[ET.Element]:
    if _local(element) in {"follow_ups", "drugs", "radiations"}:
        return
    if not len(element):
        yield element
    for child in element:
        yield from _leaves(child)


def read_records(path: str | Path) -> list[ClinicalRecord]:
    """Read a saved XML file or directory, refusing absent or ambiguous sources.

    XML procurement attributes distinguish Not Available from NO. Parsing never
    merges follow-up fields into enrollment. Duplicate patients/files or record
    identities are refused instead of arbitrarily selecting a version.
    """
    root = Path(path).expanduser()
    files = sorted(root.rglob("*.xml")) if root.is_dir() else [root]
    if not files or any(not p.is_file() or p.suffix.lower() != ".xml" for p in files):
        raise IngestError(f"BCR clinical source requires saved clinical XML files: {root}")
    records: list[ClinicalRecord] = []
    seen_patients: set[str] = set()
    for file in files:
        content = file.read_bytes()
        # The clinical schema needs neither external entities nor a DTD.
        if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
            raise IngestError(f"BCR XML must not contain DTD/entity declarations: {file}")
        try:
            tree = ET.fromstring(content)
        except ET.ParseError as exc:
            raise IngestError(f"invalid BCR XML {file}: {exc}") from exc
        patients = [e for e in tree.iter() if _local(e) == "patient"]
        if len(patients) != 1:
            raise IngestError(f"BCR XML must contain exactly one patient: {file}")
        patient = patients[0]
        ids = [e.text for e in patient if _local(e) == "bcr_patient_barcode"]
        if len(ids) != 1 or not ids[0] or not ids[0].strip():
            raise IngestError(f"missing/ambiguous bcr_patient_barcode in {file}")
        pid = ids[0].strip()
        if pid in seen_patients:
            raise IngestError(f"duplicate BCR patient XML: {pid}; select one source snapshot")
        seen_patients.add(pid)
        source = str(file.relative_to(root)) if root.is_dir() else file.name
        nodes = [patient, *[e for e in patient.iter() if _local(e) == "follow_up"]]
        file_records: list[ClinicalRecord] = []
        for index, node in enumerate(nodes):
            fields: dict[str, list[str | None]] = defaultdict(list)
            statuses: dict[str, list[str | None]] = defaultdict(list)
            for leaf in _leaves(node):
                fields[_local(leaf)].append(leaf.text)
                statuses[_local(leaf)].append(leaf.get("procurement_status"))
            record_ids = fields.get("bcr_followup_barcode", [])
            if index and (len(record_ids) != 1 or not record_ids[0]):
                raise IngestError(f"missing/ambiguous bcr_followup_barcode for {pid}")
            file_records.append(
                ClinicalRecord(
                    patient=pid,
                    record=str(record_ids[0]) if index else "enrollment",
                    kind="follow_up" if index else "enrollment",
                    fields={k: tuple(v) for k, v in fields.items()},
                    source=source,
                    checksum=_checksum(content),
                    procurement_status={k: tuple(v) for k, v in statuses.items()},
                )
            )
        present = {k for r in file_records for k in r.fields}
        required = {"biochemical_recurrence", DAY_FIELDS[0], CONTACT_FIELD}
        if missing := required - present:
            raise IngestError(f"{pid}: BCR source missing required fields {sorted(missing)}")
        records.extend(file_records)
    return records


def _positive_days(record: ClinicalRecord, field: str) -> list[float]:
    days: list[float] = []
    for raw in record.values(field):
        try:
            day = float(raw)
        except ValueError:
            continue
        if math.isfinite(day) and day > 0:
            days.append(day)
    return days


def _event_evidence(record: ClinicalRecord) -> list[str]:
    reasons = []
    if "YES" in record.values("biochemical_recurrence"):
        reasons.append("biochemical_recurrence=YES")
    for field in TYPE_FIELDS:
        if any("biochemical" in value.lower() for value in record.values(field)):
            reasons.append(field + "=biochemical")
    for field in DAY_FIELDS:
        if record.values(field):
            reasons.append(field + " populated")
    return reasons


@dataclass
class BCRReport:
    """Clinical counts and the complete ledger, including excluded patients."""

    counts: dict[str, int | None]
    patients: list[dict[str, Any]]
    sources: list[dict[str, str]]
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance for QC and saved cohort manifests."""
        return {
            "source": "TCGA-PRAD clinical supplement XML",
            "endpoint": "longitudinal biochemical recurrence",
            "derivation": "tcga_bcr_v1",
            "days_per_month": DAYS_PER_MONTH,
            "cap_months": CAP_MONTHS,
            "field_mapping": FIELD_MAPPING,
            "checksum": self.checksum,
            **self.counts,
            "sources": self.sources,
            "patients": self.patients,
        }

    def record_expression_match(self, clinical: pd.DataFrame, shared: list[str]) -> None:
        """Record matched counts separately from full clinical-source counts."""
        self.counts["n_after_expression_match"] = len(shared)
        selected = clinical.loc[shared]
        self.counts["n_matched_events_before_cap"] = int(selected["event_uncapped"].sum())
        self.counts["n_matched_events_after_cap"] = int(selected["event"].sum())
        self.counts["n_matched_censored_by_cap"] = int(
            ((selected["event_uncapped"] == 1) & (selected["event"] == 0)).sum()
        )
        matched = set(shared)
        for patient in self.patients:
            patient["matched_expression"] = patient["patient"] in matched


def derive_endpoint(records: Iterable[ClinicalRecord]) -> tuple[pd.DataFrame, BCRReport]:
    """Adjudicate every record per patient, independent of expression or CDR.

    Populated BCR-day fields establish event evidence even if the value cannot
    be used as a time. They never license fallback to a negative observation.
    ``n_unknown_excluded`` includes negatives without a usable contact date;
    finer loss counts distinguish those from entirely unknown status.
    """
    grouped: dict[str, list[ClinicalRecord]] = defaultdict(list)
    sources: dict[str, str] = {}
    identities: set[tuple[str, str]] = set()
    for record in records:
        if not record.patient or not record.record or not record.checksum:
            raise IngestError("BCR records require patient, record identity and source checksum")
        identity = (record.patient, record.record)
        if identity in identities:
            raise IngestError(f"duplicate BCR observation: {identity}")
        identities.add(identity)
        if record.source in sources and sources[record.source] != record.checksum:
            raise IngestError(f"conflicting BCR source checksums: {record.source}")
        sources[record.source] = record.checksum
        grouped[record.patient].append(record)
    if not grouped:
        raise IngestError("no BCR clinical records; no endpoint can be derived")
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    counts: dict[str, int | None] = dict.fromkeys(
        (
            "n_source",
            "n_explicit_bcr",
            "n_timed_bcr",
            "n_untimed_bcr_excluded",
            "n_supported_censored",
            "n_unknown_excluded",
            "n_no_negative_evidence",
            "n_negative_without_time",
            "n_before_expression_match",
            "n_events_before_cap",
            "n_events_after_cap",
            "n_censored_by_cap",
            "n_followup_capped",
        ),
        0,
    )
    counts["n_after_expression_match"] = None

    def increment(key: str) -> None:
        counts[key] = int(counts[key] or 0) + 1

    for pid, observations in sorted(grouped.items()):
        observations.sort(key=lambda r: (r.kind != "enrollment", r.record))
        increment("n_source")
        evidence = [
            {"record": r.record, "reasons": reasons}
            for r in observations
            if (reasons := _event_evidence(r))
        ]
        first = sorted(
            (day, r.record) for r in observations for day in _positive_days(r, DAY_FIELDS[0])
        )
        negatives = [r for r in observations if any("NO" in r.values(f) for f in NEGATIVE_FIELDS)]
        supported = sorted(
            (day, r.record) for r in negatives for day in _positive_days(r, CONTACT_FIELD)
        )
        event: int | None = None
        selected: str | None = None
        day: float | None = None
        if evidence:
            increment("n_explicit_bcr")
            if first:
                day, selected = first[0]
                event, reason = 1, "earliest_valid_first_bcr"
                increment("n_timed_bcr")
            else:
                reason = "explicit_bcr_without_valid_first_bcr_day"
                increment("n_untimed_bcr_excluded")
        elif supported:
            day, selected = supported[-1]
            event, reason = 0, "latest_supported_negative_contact"
            increment("n_supported_censored")
        else:
            reason = "negative_without_valid_contact" if negatives else "unknown_bcr_status"
            increment("n_unknown_excluded")
            increment("n_negative_without_time" if negatives else "n_no_negative_evidence")
        conflicts: list[dict[str, Any]] = []
        if evidence and negatives:
            conflicts.append(
                {
                    "kind": "positive_and_negative_observations",
                    "records": sorted(
                        {e["record"] for e in evidence} | {r.record for r in negatives}
                    ),
                }
            )
        if len({d for d, _ in first}) > 1:
            conflicts.append({"kind": "differing_first_bcr_days", "observations": first})
        months = day / DAYS_PER_MONTH if day is not None else None
        capped_event = None
        if months is not None and event is not None:
            capped_event = int(event == 1 and months <= CAP_MONTHS)
            increment("n_before_expression_match")
            if event:
                increment("n_events_before_cap")
                increment("n_events_after_cap" if capped_event else "n_censored_by_cap")
            elif months > CAP_MONTHS:
                increment("n_followup_capped")
            rows.append(
                {
                    "patient": pid,
                    "time_days": day,
                    "event_uncapped": event,
                    "time_months_uncapped": months,
                    "time_months": min(months, CAP_MONTHS),
                    "event": capped_event,
                    "selected_record": selected,
                    "reason": reason,
                }
            )
        ledger.append(
            {
                "patient": pid,
                "included": event is not None,
                "selected_record": selected,
                "selected_day": day,
                "event_uncapped": event,
                "event": capped_event,
                "reason": reason,
                "event_evidence": evidence,
                "first_bcr_candidates": first,
                "negative_candidates": supported,
                "conflicts": conflicts,
                "records": [asdict(r) for r in observations],
            }
        )
    columns = [
        "patient",
        "time_days",
        "event_uncapped",
        "time_months_uncapped",
        "time_months",
        "event",
        "selected_record",
        "reason",
    ]
    clinical = pd.DataFrame(rows, columns=columns).set_index("patient")
    # Relocating/flattening an XML directory does not change its content identity.
    digest = _checksum("\n".join(sorted(sources.values())).encode())
    report = BCRReport(
        counts, ledger, [{"path": p, "checksum": c} for p, c in sorted(sources.items())], digest
    )
    return clinical, report


def load_endpoint(
    path: str | Path,
    expected_checksum: str | None = None,
) -> tuple[pd.DataFrame, BCRReport]:
    """Load and adjudicate a saved clinical XML snapshot, optionally checksum-pinned."""
    clinical, report = derive_endpoint(read_records(path))
    if expected_checksum is not None and report.checksum != expected_checksum:
        raise IngestError(
            f"BCR clinical source checksum mismatch: expected {expected_checksum}, "
            f"found {report.checksum}; verify the source snapshot before changing its pin"
        )
    # Detect accidental non-JSON state before it enters a cohort manifest.
    json.dumps(report.to_dict(), allow_nan=False)
    return clinical, report
