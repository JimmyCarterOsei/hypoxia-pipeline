"""GDC API access for TCGA cohorts.

TCGA is not distributed as one matrix. A cohort is assembled from several
hundred per-aliquot STAR-counts files, each identified by a UUID that means
nothing on its own, plus a clinical table from a different endpoint keyed on a
different identifier. This module builds the file manifest, fetches the files,
and returns the pieces the ingest pipeline joins.

The manifest is the important artefact. It records, for one query against one
GDC release, exactly which file UUIDs were selected and which aliquot each
belongs to. Two consequences:

* the query is reproducible - a later run either gets the same UUIDs or the
  manifest comparison says the release moved;
* every file is cached and hashed individually, so provenance is per file
  rather than per cohort.

Only :func:`fetch_manifest`, :func:`fetch_file` and :func:`fetch_clinical`
touch the network, and all three go through the cache. Everything else is a
pure function tested offline.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd

from hypoxiapipe.errors import CacheMissError, DownloadError, IngestError, ParseError
from hypoxiapipe.ingest.cache import Cache, CacheEntry
from hypoxiapipe.ingest.tcga import parse_barcode, parse_star_counts

GDC_API = "https://api.gdc.cancer.gov"
FILES_ENDPOINT = f"{GDC_API}/files"
CASES_ENDPOINT = f"{GDC_API}/cases"
DATA_ENDPOINT = f"{GDC_API}/data"

STAR_WORKFLOW = "STAR - Counts"
EXPRESSION_DATA_TYPE = "Gene Expression Quantification"

#: Fields requested for each expression file. The aliquot submitter_id is the
#: TCGA barcode; without it a file UUID cannot be tied to a patient.
FILE_FIELDS = (
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "data_type",
    "analysis.workflow_type",
    "cases.samples.sample_type",
    "cases.samples.portions.analytes.aliquots.submitter_id",
    "cases.submitter_id",
)

CLINICAL_FIELDS = (
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.primary_gleason_grade",
    "diagnoses.secondary_gleason_grade",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
)


def _ssl_context() -> ssl.SSLContext:
    """Return a verifying SSL context backed by certifi's CA bundle.

    Python installations that are not wired into the system trust store - a
    Homebrew or python.org build on macOS is the usual case - otherwise fail
    every HTTPS request with a certificate error that looks like a bug here.
    """
    import certifi  # noqa: PLC0415 - keeps import cost off the module path

    return ssl.create_default_context(cafile=certifi.where())


def _post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> bytes:
    """POST a JSON body and return the raw response bytes."""
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=timeout, context=_ssl_context()
    ) as response:
        return bytes(response.read())


def manifest_query(
    project: str,
    workflow: str = STAR_WORKFLOW,
    data_type: str = EXPRESSION_DATA_TYPE,
    size: int = 5000,
) -> dict[str, Any]:
    """Build the GDC files query for one project's expression files."""
    return {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
                {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
                {"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow]}},
            ],
        },
        "fields": ",".join(FILE_FIELDS),
        "format": "JSON",
        "size": size,
    }


def parse_manifest(payload: str | bytes) -> pd.DataFrame:
    """Parse a GDC files response into a tidy manifest.

    One row per file, carrying the UUID, the aliquot barcode, the patient and
    the sample type. Files whose barcode cannot be resolved are an error rather
    than a silent drop: a file that cannot be tied to a patient would otherwise
    vanish between the manifest and the matrix.
    """
    raw = json.loads(payload)
    hits = raw.get("data", {}).get("hits", raw.get("hits", []))
    if not hits:
        raise ParseError("GDC returned no files for this query")

    rows: list[dict[str, Any]] = []
    for hit in hits:
        cases = hit.get("cases") or []
        if not cases:
            raise ParseError(f"file {hit.get('file_id')} has no case attached")
        samples = cases[0].get("samples") or []
        barcode = None
        sample_type = None
        if samples:
            sample_type = samples[0].get("sample_type")
            for portion in samples[0].get("portions") or []:
                for analyte in portion.get("analytes") or []:
                    for aliquot in analyte.get("aliquots") or []:
                        barcode = aliquot.get("submitter_id")
                        break
        if not barcode:
            raise ParseError(
                f"file {hit.get('file_id')} has no aliquot barcode; the query must request "
                "cases.samples.portions.analytes.aliquots.submitter_id"
            )
        rows.append(
            {
                "file_id": hit.get("file_id"),
                "file_name": hit.get("file_name"),
                "file_size": hit.get("file_size"),
                "md5sum": hit.get("md5sum"),
                "barcode": barcode,
                "patient": parse_barcode(barcode).patient,
                "sample_type": sample_type,
                "workflow": (hit.get("analysis") or {}).get("workflow_type"),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("barcode").reset_index(drop=True)
    if manifest["file_id"].duplicated().any():
        raise ParseError("GDC returned duplicate file UUIDs")
    return manifest


def fetch_manifest(
    project: str,
    cache: Cache,
    workflow: str = STAR_WORKFLOW,
    refresh: bool = False,
) -> tuple[pd.DataFrame, CacheEntry]:
    """Fetch (or reuse) the file manifest for a project.

    The response is cached under a key naming the project and workflow, so a
    later run reuses the same file selection instead of silently picking up a
    new GDC release mid-analysis. Pass ``refresh=True`` to re-query on purpose.
    """
    key = f"gdc/{project}/{workflow.replace(' ', '_')}_manifest.json"
    if refresh and cache.has(key):
        cache.path_for(key).unlink()

    query = manifest_query(project, workflow=workflow)

    def download(url: str) -> bytes:
        return _post_json(url, query)

    entry = cache.fetch(key, FILES_ENDPOINT, downloader=download)
    return parse_manifest(entry.path.read_bytes()), entry


def fetch_file(file_id: str, cache: Cache) -> CacheEntry:
    """Fetch one GDC data file by UUID, caching and hashing it individually."""
    return cache.fetch(f"gdc/files/{file_id}", f"{DATA_ENDPOINT}/{file_id}")


def parse_clinical(payload: str | bytes) -> pd.DataFrame:
    """Parse a GDC cases response into a patient-indexed clinical table."""
    raw = json.loads(payload)
    hits = raw.get("data", {}).get("hits", raw.get("hits", []))
    if not hits:
        raise ParseError("GDC returned no cases for this query")

    rows: list[dict[str, Any]] = []
    for hit in hits:
        demographic = hit.get("demographic") or {}
        diagnoses = hit.get("diagnoses") or [{}]
        diagnosis = diagnoses[0] if diagnoses else {}
        rows.append(
            {
                "patient": hit.get("submitter_id"),
                "vital_status": demographic.get("vital_status"),
                "days_to_death": demographic.get("days_to_death"),
                "age_at_index": demographic.get("age_at_index"),
                "days_to_last_follow_up": diagnosis.get("days_to_last_follow_up"),
                "primary_gleason": diagnosis.get("primary_gleason_grade"),
                "secondary_gleason": diagnosis.get("secondary_gleason_grade"),
                "ajcc_t": diagnosis.get("ajcc_pathologic_t"),
                "ajcc_n": diagnosis.get("ajcc_pathologic_n"),
            }
        )
    clinical = pd.DataFrame(rows).set_index("patient")
    clinical.index = clinical.index.astype(str)
    return clinical[~clinical.index.duplicated(keep="first")]


def fetch_clinical(project: str, cache: Cache, size: int = 5000) -> tuple[pd.DataFrame, CacheEntry]:
    """Fetch (or reuse) the clinical table for a project."""
    key = f"gdc/{project}/clinical.json"
    query = {
        "filters": {
            "op": "in",
            "content": {"field": "project.project_id", "value": [project]},
        },
        "fields": ",".join(CLINICAL_FIELDS),
        "format": "JSON",
        "size": size,
    }

    def download(url: str) -> bytes:
        return _post_json(url, query)

    entry = cache.fetch(key, CASES_ENDPOINT, downloader=download)
    return parse_clinical(entry.path.read_bytes()), entry


@dataclass(frozen=True)
class DownloadReport:
    """What assembling a matrix from per-file downloads actually fetched."""

    n_files: int
    n_parsed: int
    n_failed: int
    failures: tuple[str, ...] = ()
    value_column: str = "tpm_unstranded"

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable summary for the QC report and manifest."""
        return {
            "n_files": self.n_files,
            "n_parsed": self.n_parsed,
            "n_failed": self.n_failed,
            "failures": list(self.failures[:10]),
            "value_column": self.value_column,
        }


def load_expression(
    manifest: pd.DataFrame,
    cache: Cache,
    value_column: str = "tpm_unstranded",
    tolerate_failures: int = 0,
) -> tuple[dict[str, pd.Series], DownloadReport]:
    """Fetch and parse every file in a manifest into per-aliquot series.

    ``tolerate_failures`` is zero by default. A cohort silently missing a
    handful of samples is a different cohort from the one the manifest
    describes, and the difference is invisible once the matrix is built.
    """
    per_sample: dict[str, pd.Series] = {}
    failures: list[str] = []

    for row in manifest.itertuples():
        try:
            entry = fetch_file(str(row.file_id), cache)
            text = entry.path.read_text(errors="replace")
            per_sample[str(row.barcode)] = parse_star_counts(text, value_column=value_column)
        except (CacheMissError, DownloadError, ParseError, OSError) as exc:
            # A cache miss is a per-file failure like any other, so the
            # tolerate_failures policy applies identically online and offline.
            failures.append(f"{row.barcode} ({row.file_id}): {exc}")

    report = DownloadReport(
        n_files=int(manifest.shape[0]),
        n_parsed=len(per_sample),
        n_failed=len(failures),
        failures=tuple(failures),
        value_column=value_column,
    )
    if failures and len(failures) > tolerate_failures:
        raise IngestError(
            f"{len(failures)} of {manifest.shape[0]} files failed and "
            f"tolerate_failures={tolerate_failures}:\n  " + "\n  ".join(failures[:5])
        )
    if not per_sample:
        raise IngestError("no expression files could be parsed")
    return per_sample, report
