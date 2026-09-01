"""TCGA survival endpoints from the Clinical Data Resource.

The GDC clinical endpoint does not carry a usable recurrence outcome for most
projects, and for TCGA-PRAD in particular it carries almost nothing usable at
all: prostate cancer patients in TCGA rarely die during follow-up, so overall
survival has roughly a dozen events across nearly 500 patients. Fitting a Cox
model on that is not a conservative analysis, it is an empty one.

The standard fix is the TCGA Clinical Data Resource (Liu et al., *Cell* 2018),
which curates four harmonised endpoints - OS, DSS, DFI and PFI - and, crucially,
states which are usable per tumour type. For PRAD the recommendation is PFI:
progression-free interval, which for a surgically treated prostate cohort is
driven by biochemical recurrence.

This module refuses to build an endpoint the CDR authors marked unusable for
that tumour type. That is deliberate. The alternative - letting a caller ask
for OS in PRAD and returning a model with eleven events - produces a hazard
ratio, a confidence interval and a p-value that all look like results.

The CDR table is not redistributed here. It is a supplementary file from the
paper; point ``clinical_path`` at your own copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hypoxiapipe.errors import EndpointError, IngestError

#: CDR endpoint -> (event column, time column) in the published table.
CDR_ENDPOINTS = {
    "OS": ("OS", "OS.time"),
    "DSS": ("DSS", "DSS.time"),
    "DFI": ("DFI", "DFI.time"),
    "PFI": ("PFI", "PFI.time"),
}

#: Endpoints Liu et al. recommend against for a given tumour type, with the
#: reason. Not exhaustive - it covers the projects this pipeline uses, and an
#: unlisted project is simply unchecked rather than assumed fine.
DISCOURAGED: dict[str, dict[str, str]] = {
    "PRAD": {
        "OS": "too few deaths in TCGA-PRAD for a meaningful survival model",
        "DSS": "too few disease-specific deaths in TCGA-PRAD",
        "DFI": "DFI is sparsely recorded in PRAD; PFI is the recommended endpoint",
    },
    "THCA": {
        "OS": "too few deaths in TCGA-THCA",
        "DSS": "too few disease-specific deaths in TCGA-THCA",
    },
    "LGG": {"DFI": "DFI is not recommended for LGG"},
}

#: The CDR ships as an Excel supplement; a TSV/CSV export is equally acceptable.
READERS = {
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".txt": lambda p: pd.read_csv(p, sep="\t"),
    ".csv": lambda p: pd.read_csv(p),
}


@dataclass(frozen=True)
class CDRReport:
    """What the CDR gave for one project and endpoint."""

    project: str
    endpoint: str
    n_rows: int
    n_project: int
    n_usable: int
    n_events: int
    cap_months: float | None = None
    n_censored_by_cap: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable summary for the QC report and manifest."""
        return {
            "source": "TCGA-CDR",
            "project": self.project,
            "endpoint": self.endpoint,
            "n_rows": self.n_rows,
            "n_project": self.n_project,
            "n_usable": self.n_usable,
            "n_events": self.n_events,
            "cap_months": self.cap_months,
            "n_censored_by_cap": self.n_censored_by_cap,
        }


def check_endpoint(project: str, endpoint: str) -> None:
    """Raise if the CDR authors advise against this endpoint for this project."""
    if endpoint not in CDR_ENDPOINTS:
        raise EndpointError(
            f"unknown CDR endpoint {endpoint!r} (choose from {sorted(CDR_ENDPOINTS)})"
        )
    short = project.replace("TCGA-", "").upper()
    reason = DISCOURAGED.get(short, {}).get(endpoint.upper())
    if reason:
        recommended = "PFI" if short in {"PRAD", "THCA"} else "PFI or OS"
        raise EndpointError(
            f"{endpoint} is not a usable endpoint for {project}: {reason}. "
            f"Liu et al. (Cell 2018) recommend {recommended}. If you intend to override "
            "this, do it explicitly in code rather than by passing a different string."
        )


def read_cdr(path: str | Path) -> pd.DataFrame:
    """Read a TCGA-CDR table from Excel, TSV or CSV."""
    p = Path(path)
    if not p.exists():
        raise IngestError(
            f"TCGA-CDR table not found at {p}. It is a supplementary file from "
            "Liu et al. (Cell 2018) and is not redistributed with this package; "
            "download it and point clinical_path at your copy."
        )
    reader = READERS.get(p.suffix.lower())
    if reader is None:
        raise IngestError(f"unsupported CDR file type {p.suffix!r} (expected {sorted(READERS)})")
    table = reader(p)
    if "bcr_patient_barcode" not in table.columns:
        raise IngestError(
            f"{p} does not look like the TCGA-CDR table: no 'bcr_patient_barcode' column "
            f"(found {list(table.columns)[:8]})"
        )
    return table


def load_endpoint(
    path: str | Path,
    project: str,
    endpoint: str = "PFI",
    cap_months: float | None = 60.0,
    allow_discouraged: bool = False,
    time_out: str = "time_months",
    event_out: str = "event",
) -> tuple[pd.DataFrame, CDRReport]:
    """Build a patient-indexed clinical table with a usable survival endpoint.

    Parameters
    ----------
    path : str | Path
        Location of the TCGA-CDR table.
    project : str
        Project ID, e.g. ``TCGA-PRAD``; used to subset and to check the endpoint.
    endpoint : str
        One of OS, DSS, DFI, PFI.
    cap_months : float | None
        Administrative censoring horizon; events beyond it become censored.
    allow_discouraged : bool
        Override the per-tumour endpoint check. Requires a deliberate call.
    time_out : str
        Name of the derived time column.
    event_out : str
        Name of the derived event column.

    """
    if not allow_discouraged:
        check_endpoint(project, endpoint)

    table = read_cdr(path)
    event_col, time_col = CDR_ENDPOINTS[endpoint]
    for column in (event_col, time_col):
        if column not in table.columns:
            raise EndpointError(f"CDR table has no '{column}' column")

    n_rows = int(table.shape[0])
    if "type" in table.columns:
        short = project.replace("TCGA-", "").upper()
        table = table[table["type"].astype(str).str.upper() == short]
    if table.empty:
        raise EndpointError(f"no {project} rows in the CDR table")

    clinical = table.set_index(table["bcr_patient_barcode"].astype(str))
    clinical.index.name = "patient"
    clinical = clinical[~clinical.index.duplicated(keep="first")]

    # CDR times are in days; the rest of the pipeline works in months.
    time = pd.to_numeric(clinical[time_col], errors="coerce") / 30.4375
    event = pd.to_numeric(clinical[event_col], errors="coerce")

    n_capped = 0
    if cap_months is not None:
        beyond = (time > cap_months) & time.notna()
        n_capped = int((beyond & (event == 1)).sum())
        event = event.where(~beyond, 0.0)
        time = time.clip(upper=cap_months)

    out = clinical.copy()
    out[time_out] = time
    out[event_out] = event
    usable = time.notna() & event.notna() & (time > 0)

    report = CDRReport(
        project=project,
        endpoint=endpoint,
        n_rows=n_rows,
        n_project=int(clinical.shape[0]),
        n_usable=int(usable.sum()),
        n_events=int(event[usable].sum()),
        cap_months=cap_months,
        n_censored_by_cap=n_capped,
    )
    return out, report
