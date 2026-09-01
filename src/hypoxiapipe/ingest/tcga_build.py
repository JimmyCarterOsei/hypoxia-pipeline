"""Assembling a TCGA cohort from GDC parts.

:func:`load_tcga` is the counterpart to ``load_geo``: it turns a project ID
into a :class:`Cohort` with an endpoint attached. The steps are individually
tested elsewhere; the value of having them in one ordered place is that the
joins happen in the only order that is correct.

Three joins can each silently produce an empty or wrong cohort:

* **Aliquot to patient.** Expression is per aliquot
  (``TCGA-CH-5761-01A-11R-1580-07``), clinical is per patient
  (``TCGA-CH-5761``). Joining without truncating gives an empty intersection.
* **Tumour versus normal.** Adjacent normals are in the same project. Leaving
  them in a prognostic cohort widens every gene's apparent dynamic range and
  adds patients with no meaningful outcome.
* **Patient with several aliquots.** Truncating without deduplicating produces
  duplicate patient columns, which then get silently dropped or averaged by
  whatever runs next.

Each is handled explicitly and counted in the returned report, so the number of
samples lost at every stage is visible rather than inferred from a final total.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from hypoxiapipe.errors import IngestError
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.cdr import check_endpoint, load_endpoint, read_cdr
from hypoxiapipe.ingest.cohort import Cohort, Provenance
from hypoxiapipe.ingest.gdc import fetch_clinical, fetch_manifest, load_expression
from hypoxiapipe.ingest.spec import CohortSpec
from hypoxiapipe.ingest.tcga import (
    assemble_matrix,
    restrict_to_primary_tumours,
    to_patient_level,
)

TIME_COLUMN = "time_months"
EVENT_COLUMN = "event"


def _gdc_overall_survival(clinical: pd.DataFrame) -> pd.DataFrame:
    """Derive overall survival from GDC clinical fields.

    Provided for projects where OS is a usable endpoint. For PRAD it is not,
    which is why the spec's default clinical source is the CDR - see
    :mod:`hypoxiapipe.ingest.cdr`.
    """
    required = ("vital_status", "days_to_death", "days_to_last_follow_up")
    missing = [c for c in required if c not in clinical.columns]
    if missing:
        raise IngestError(
            f"GDC clinical table is missing {missing}; overall survival cannot be derived. "
            "Request these fields in the cases query, or use clinical_source: cdr."
        )

    days_to_death = pd.to_numeric(clinical["days_to_death"], errors="coerce")
    days_to_last = pd.to_numeric(clinical["days_to_last_follow_up"], errors="coerce")
    event = (clinical["vital_status"].astype(str).str.lower() == "dead").astype(float)
    time = days_to_death.where(event == 1, days_to_last) / 30.4375

    out = clinical.copy()
    out[TIME_COLUMN] = time
    out[EVENT_COLUMN] = event
    return out


def load_tcga(
    spec: CohortSpec,
    cache: Cache,
    min_samples: int = 30,
) -> tuple[Cohort, dict[str, Any]]:
    """Build a TCGA cohort: manifest, expression, sample selection, endpoint.

    Parameters
    ----------
    spec : CohortSpec
        Cohort spec with ``source: tcga`` and an ``accession`` naming the project.
    cache : Cache
        Content-addressed cache; every GDC file is fetched and hashed through it.
    min_samples : int
        Floor below which the assembled cohort is not worth returning.

    """
    project = str(spec.accession)
    report: dict[str, Any] = {"project": project, "workflow": spec.workflow}

    # Check the clinical source before fetching ~2 GB of expression files. The
    # join happens last, so an unreadable CDR path would otherwise surface at
    # the end of a long download rather than at the start of one.
    if spec.clinical_source == "cdr":
        check_endpoint(project, spec.cdr_endpoint)
        read_cdr(str(spec.clinical_path))

    # -- manifest ---------------------------------------------------------
    manifest, manifest_entry = fetch_manifest(project, cache, workflow=spec.workflow)
    report["manifest"] = {
        "n_files": int(manifest.shape[0]),
        "checksum": manifest_entry.checksum,
        "retrieved_at": manifest_entry.retrieved_at,
        "n_patients_in_manifest": int(manifest["patient"].nunique()),
    }

    # -- expression -------------------------------------------------------
    per_sample, download_report = load_expression(
        manifest,
        cache,
        value_column=spec.star_value_column,
        tolerate_failures=spec.tolerate_file_failures,
    )
    report["download"] = download_report.to_dict()

    expr = assemble_matrix(per_sample, collapse=spec.collapse_rule)
    report["assembled"] = {"n_genes": int(expr.shape[0]), "n_aliquots": int(expr.shape[1])}

    # -- sample selection -------------------------------------------------
    dropped_types: list[str] = []
    if spec.primary_tumours_only:
        expr, dropped_types = restrict_to_primary_tumours(expr)
    expr, duplicated_patients = to_patient_level(expr, rule=spec.duplicate_aliquot_rule)
    report["selection"] = {
        "n_dropped_non_primary": len(dropped_types),
        "dropped_non_primary": dropped_types[:10],
        "n_patients_with_multiple_aliquots": len(duplicated_patients),
        "duplicate_aliquot_rule": spec.duplicate_aliquot_rule,
        "n_patients": int(expr.shape[1]),
    }

    # STAR TPM is linear and the rest of the pipeline expects a log scale, but
    # the transform is applied by build_cohort's scale stage, not here. Doing it
    # in both places log2s the matrix twice, which is silent: the values stay
    # positive and monotonic, so nothing errors and every downstream number is
    # computed on log2(log2(TPM + 1) + 1). The only visible symptom is a value
    # range that tops out around 4 instead of ~15.

    # -- clinical + endpoint ----------------------------------------------
    if spec.clinical_source == "cdr":
        clinical, cdr_report = load_endpoint(
            str(spec.clinical_path),
            project,
            endpoint=spec.cdr_endpoint,
            cap_months=spec.endpoint.cap_months if spec.endpoint else 60.0,
        )
        report["endpoint"] = cdr_report.to_dict()
        clinical_source_detail = {"clinical_source": "TCGA-CDR", "path": str(spec.clinical_path)}
    else:
        clinical, clinical_entry = fetch_clinical(project, cache)
        clinical = _gdc_overall_survival(clinical)
        report["endpoint"] = {
            "source": "GDC clinical",
            "endpoint": "OS",
            "n_patients": int(clinical.shape[0]),
            "n_events": int(clinical[EVENT_COLUMN].fillna(0).sum()),
        }
        clinical_source_detail = {
            "clinical_source": "GDC",
            "checksum": clinical_entry.checksum,
        }

    shared = [p for p in expr.columns if p in clinical.index]
    if not shared:
        raise IngestError(
            f"{project}: no patients in common between expression and clinical data. "
            f"Expression columns look like {list(expr.columns)[:2]}; clinical index looks "
            f"like {list(clinical.index)[:2]}. Both must be patient-level TCGA barcodes."
        )
    report["join"] = {
        "n_expression_patients": int(expr.shape[1]),
        "n_clinical_patients": int(clinical.shape[0]),
        "n_joined": len(shared),
        "n_expression_without_clinical": int(expr.shape[1]) - len(shared),
    }

    provenance = (
        Provenance(
            source="tcga",
            accession=project,
            url="https://api.gdc.cancer.gov",
            platform=spec.workflow,
        )
        .with_step(
            "gdc_manifest",
            n_files=int(manifest.shape[0]),
            checksum=manifest_entry.checksum,
            workflow=spec.workflow,
        )
        .with_step("load_star_counts", **download_report.to_dict())
        .with_step("select_samples", **report["selection"])
        .with_step("clinical", **clinical_source_detail, **report["endpoint"])
    )

    cohort = Cohort.align(
        name=spec.name,
        expr=expr,
        clinical=clinical,
        provenance=provenance,
        min_samples=min_samples,
    )
    return cohort, report
