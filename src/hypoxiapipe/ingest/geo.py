"""GEO series-matrix ingest.

The parsing is deliberately separated from the download: ``parse_series_matrix``
is a pure function over text, so it is tested offline against small committed
fixtures, while ``fetch_series_matrix`` only handles bytes and caching.

Two GEO facts this module encodes, both learned the hard way:

* Sample characteristics arrive as repeated ``!Sample_characteristics_ch1``
  rows of ``key: value`` pairs, in inconsistent order across samples. They are
  parsed per sample into a wide clinical frame rather than assumed positional.
* The matrix is indexed by *platform probe IDs*, not gene symbols. Mapping
  probes to symbols requires the GPL annotation and is a separate step, so this
  module reports the platform rather than pretending the index is genes.
"""

from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass

import pandas as pd

from hypoxiapipe.errors import ParseError
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.cohort import Cohort, Provenance

GEO_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/matrix/{acc}_series_matrix.txt.gz"
)

_CHAR_KV = re.compile(r"^\s*([^:]+?)\s*:\s*(.*)$")


def series_matrix_url(accession: str) -> str:
    """Canonical GEO series-matrix URL for an accession such as 'GSE70768'."""
    if not re.fullmatch(r"GSE\d+", accession):
        raise ValueError(f"not a GEO series accession: {accession!r}")
    stub = accession[:-3] + "nnn"
    return GEO_MATRIX_URL.format(stub=stub, acc=accession)


@dataclass(frozen=True)
class SeriesMatrix:
    """Parsed contents of a GEO series-matrix file."""

    expr: pd.DataFrame
    clinical: pd.DataFrame
    platform: str | None
    title: str | None


def _clean(value: str) -> str:
    return value.strip().strip('"')


def parse_series_matrix(text: str) -> SeriesMatrix:
    """Parse GEO series-matrix text into an expression matrix and clinical frame."""
    meta: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    table_lines: list[str] = []
    in_table = False

    for line in text.splitlines():
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            in_table = False
            continue
        if in_table:
            if line.strip():
                table_lines.append(line)
            continue
        if line.startswith("!Sample_characteristics_ch"):
            characteristics.append([_clean(v) for v in line.split("\t")[1:]])
        elif line.startswith("!"):
            parts = line.split("\t")
            meta[parts[0].lstrip("!")] = [_clean(v) for v in parts[1:]]

    if not table_lines:
        raise ParseError("no expression table found (missing series_matrix_table_begin block)")

    expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0)
    expr.index = expr.index.map(lambda x: str(x).strip('"'))
    expr.columns = [str(c).strip('"') for c in expr.columns]
    expr = expr.apply(pd.to_numeric, errors="coerce")

    sample_ids = meta.get("Sample_geo_accession", list(expr.columns))
    clinical = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))

    for row in characteristics:
        if len(row) != len(clinical.index):
            continue
        keys = [m.group(1).lower().replace(" ", "_") for v in row if (m := _CHAR_KV.match(v))]
        if not keys:
            continue
        key = max(set(keys), key=keys.count)
        values = [m.group(2) if (m := _CHAR_KV.match(v)) else None for v in row]
        column = key if key not in clinical.columns else f"{key}_{len(clinical.columns)}"
        clinical[column] = values

    for field in ("Sample_title", "Sample_source_name_ch1"):
        if field in meta and len(meta[field]) == len(clinical.index):
            clinical[field.replace("Sample_", "").lower()] = meta[field]

    if list(expr.columns) != list(clinical.index) and len(expr.columns) == len(clinical.index):
        expr.columns = list(clinical.index)

    platform_values = meta.get("Series_platform_id") or meta.get("Sample_platform_id") or []
    title_values = meta.get("Series_title") or []
    platform = platform_values[0] if platform_values else None
    title = title_values[0] if title_values else None
    return SeriesMatrix(expr=expr, clinical=clinical, platform=platform, title=title)


def fetch_series_matrix(accession: str, cache: Cache) -> tuple[str, str, str]:
    """Return (text, url, checksum) for a series matrix, downloading once if needed."""
    url = series_matrix_url(accession)
    entry = cache.fetch(f"geo/{accession}_series_matrix.txt.gz", url)
    raw = entry.path.read_bytes()
    text = (
        gzip.decompress(raw).decode("utf-8", errors="replace")
        if raw[:2] == b"\x1f\x8b"
        else raw.decode("utf-8", errors="replace")
    )
    return text, url, entry.checksum


def load_geo(
    accession: str,
    cache: Cache,
    name: str | None = None,
    min_samples: int = 30,
) -> Cohort:
    """Load a GEO series as an aligned :class:`Cohort`."""
    text, url, checksum = fetch_series_matrix(accession, cache)
    parsed = parse_series_matrix(text)
    prov = Provenance(
        source="GEO",
        accession=accession,
        url=url,
        platform=parsed.platform,
        retrieved_at=cache.get(f"geo/{accession}_series_matrix.txt.gz").retrieved_at,
    ).with_step("download", checksum=checksum, title=parsed.title)
    return Cohort.align(
        name=name or accession,
        expr=parsed.expr,
        clinical=parsed.clinical,
        provenance=prov,
        min_samples=min_samples,
    )
