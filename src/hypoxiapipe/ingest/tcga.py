"""TCGA ingest via the GDC API.

As with GEO, parsing and assembly are pure functions tested offline; only
``fetch_star_counts`` touches the network.

Three TCGA-specific traps this module handles explicitly:

* **Ensembl versioning.** STAR-counts files carry ``ENSG00000141510.16``. The
  version suffix is stripped, because it changes between GDC data releases and
  silently breaks joins against anything pinned to an earlier release.
* **Barcode truncation.** Expression is per *aliquot*
  (``TCGA-CH-5761-01A-11R-1580-07``); clinical data is per *patient*
  (``TCGA-CH-5761``). Joining without truncating gives an empty intersection;
  truncating without deduplicating gives duplicate patients.
* **Sample type codes.** Positions 14-15 of the barcode encode tumour ("01")
  versus normal ("11"). Leaving normals in a prognostic cohort inflates the
  apparent dynamic range of every gene.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hypoxiapipe.errors import ParseError

BARCODE = re.compile(r"^(TCGA-[0-9A-Z]{2}-[0-9A-Z]{4})(?:-(\d{2})[A-Z]?)?")

PRIMARY_TUMOUR = "01"
STAR_VALUE_COLUMNS = (
    "unstranded",
    "stranded_first",
    "stranded_second",
    "tpm_unstranded",
    "fpkm_unstranded",
    "fpkm_uq_unstranded",
)


def strip_ensembl_version(gene_id: str) -> str:
    """Drop the ``.N`` version suffix from an Ensembl gene ID."""
    return str(gene_id).split(".", 1)[0]


@dataclass(frozen=True)
class Barcode:
    """A parsed TCGA barcode."""

    patient: str
    sample_type: str | None

    @property
    def is_primary_tumour(self) -> bool:
        """True for sample type 01 (primary solid tumour)."""
        return self.sample_type == PRIMARY_TUMOUR


def parse_barcode(barcode: str) -> Barcode:
    """Parse a TCGA barcode into patient ID and sample type code."""
    m = BARCODE.match(str(barcode).upper())
    if not m:
        raise ParseError(f"not a TCGA barcode: {barcode!r}")
    return Barcode(patient=m.group(1), sample_type=m.group(2))


def parse_star_counts(text: str, value_column: str = "tpm_unstranded") -> pd.Series:
    """Parse one GDC STAR-counts file into a gene-level series.

    The file has four leading ``N_`` summary rows (unmapped, multimapping, ...)
    which are not genes and are dropped.
    """
    if value_column not in STAR_VALUE_COLUMNS:
        raise ValueError(
            f"unknown STAR value column {value_column!r} (choose from {STAR_VALUE_COLUMNS})"
        )
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")
    required = {"gene_id", "gene_name", value_column}
    if not required.issubset(df.columns):
        raise ParseError(
            f"STAR-counts file missing columns {sorted(required - set(df.columns))}; "
            f"found {list(df.columns)[:8]}"
        )
    df = df[~df["gene_id"].astype(str).str.startswith("N_")].copy()
    df["gene_id"] = df["gene_id"].map(strip_ensembl_version)
    series = pd.Series(
        pd.to_numeric(df[value_column], errors="coerce").to_numpy(),
        index=pd.Index(df["gene_name"].astype(str), name="gene"),
    )
    return series


def assemble_matrix(per_sample: dict[str, pd.Series], collapse: str = "max_mean") -> pd.DataFrame:
    """Combine per-sample gene series into one genes x samples matrix.

    Ensembl IDs map many-to-one onto symbols, so duplicates are collapsed with
    the same rule used elsewhere in the pipeline (see ``harmonise.symbols``).
    """
    if not per_sample:
        raise ParseError("no samples to assemble")
    frame = pd.DataFrame(per_sample)
    if frame.index.duplicated().any():
        from hypoxiapipe.harmonise.symbols import collapse_duplicate_rows

        frame, _ = collapse_duplicate_rows(frame, rule=collapse)
    return frame.sort_index()


def restrict_to_primary_tumours(expr: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep only primary-tumour aliquots; return the matrix and what was dropped."""
    keep: list[str] = []
    dropped: list[str] = []
    for col in expr.columns:
        try:
            bc = parse_barcode(str(col))
        except ParseError:
            dropped.append(str(col))
            continue
        (keep if bc.is_primary_tumour or bc.sample_type is None else dropped).append(str(col))
    return expr[keep], dropped


def to_patient_level(expr: pd.DataFrame, rule: str = "first") -> tuple[pd.DataFrame, list[str]]:
    """Collapse aliquot columns to patient IDs.

    A patient with several aliquots is resolved by `rule` ("first" keeps the
    first barcode in sorted order, "mean" averages them); the affected patients
    are returned so the choice is reported rather than hidden.
    """
    patients = [parse_barcode(str(c)).patient for c in expr.columns]
    out = expr.copy()
    out.columns = patients
    duplicated = sorted({p for p in patients if patients.count(p) > 1})
    if duplicated:
        if rule == "mean":
            out = out.T.groupby(level=0).mean().T
        elif rule == "first":
            out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
        else:
            raise ValueError(f"unknown duplicate-aliquot rule {rule!r}")
    return out, duplicated


def log2_transform(expr: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """log2(x + pseudocount) for TPM/FPKM matrices on the linear scale."""
    return pd.DataFrame(
        np.log2(expr.to_numpy(dtype=float) + pseudocount),
        index=expr.index,
        columns=expr.columns,
    )
