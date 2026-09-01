"""Persisting a built cohort.

A built cohort is written as a directory rather than a single blob so that each
part is independently inspectable: two tables a human can open, and the JSON
that says where they came from.

The checksum recorded at save time is re-verified on load. A cohort whose
matrix has been edited on disk between the build and the analysis fails to load
rather than analysing quietly - the same rule the signature registry applies to
gene lists, applied to data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hypoxiapipe.errors import IngestError
from hypoxiapipe.ingest.cohort import Cohort, Provenance, ProvenanceStep

EXPR_FILE = "expression.parquet"
CLINICAL_FILE = "clinical.parquet"
META_FILE = "cohort.json"


def _provenance_from_dict(raw: dict[str, Any]) -> Provenance:
    steps = tuple(
        ProvenanceStep(
            action=s.get("action", "unknown"),
            at=s.get("at", ""),
            detail={k: v for k, v in s.items() if k not in {"action", "at"}},
        )
        for s in raw.get("steps", [])
    )
    return Provenance(
        source=raw.get("source", "unknown"),
        accession=raw.get("accession"),
        url=raw.get("url"),
        platform=raw.get("platform"),
        retrieved_at=raw.get("retrieved_at"),
        symbol_authority=raw.get("symbol_authority"),
        steps=steps,
    )


def save_cohort(cohort: Cohort, directory: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write a cohort to ``directory`` and return the path."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    cohort.expr.to_parquet(out / EXPR_FILE)
    # Clinical columns are heterogeneous (strings beside derived numerics), so
    # normalise object columns to string before writing to keep parquet happy.
    clinical = cohort.clinical.copy()
    for col in clinical.columns:
        if clinical[col].dtype == object:
            clinical[col] = clinical[col].astype("string")
    clinical.to_parquet(out / CLINICAL_FILE)

    meta: dict[str, Any] = {
        "name": cohort.name,
        "n_genes": cohort.n_genes,
        "n_samples": cohort.n_samples,
        "population_hash": cohort.population_hash,
        "expr_checksum": cohort.expr_checksum,
        "provenance": cohort.provenance.to_dict(),
    }
    if extra:
        meta.update(extra)
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return out


def load_cohort(directory: str | Path, verify: bool = True) -> Cohort:
    """Load a cohort previously written by :func:`save_cohort`.

    With ``verify=True`` the expression matrix is re-hashed and compared with
    the checksum recorded at save time.
    """
    src = Path(directory)
    for required in (EXPR_FILE, CLINICAL_FILE, META_FILE):
        if not (src / required).exists():
            raise IngestError(f"{src}: not a cohort directory (missing {required})")

    meta = json.loads((src / META_FILE).read_text())
    expr = pd.read_parquet(src / EXPR_FILE)
    clinical = pd.read_parquet(src / CLINICAL_FILE)
    clinical.index = clinical.index.astype(str)
    expr.columns = [str(c) for c in expr.columns]

    cohort = Cohort(
        name=meta.get("name", src.name),
        expr=expr,
        clinical=clinical,
        provenance=_provenance_from_dict(meta.get("provenance", {})),
    )

    recorded = meta.get("expr_checksum")
    if verify and recorded and recorded != cohort.expr_checksum:
        raise IngestError(
            f"{src}: expression checksum mismatch.\n"
            f"  recorded: {recorded}\n  actual:   {cohort.expr_checksum}\n"
            "The stored matrix has changed since it was built. Rebuild the cohort "
            "rather than analysing a file of unknown provenance."
        )
    return cohort
