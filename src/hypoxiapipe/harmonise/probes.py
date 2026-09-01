"""Probe-to-symbol mapping for array platforms.

A GEO series matrix is indexed by *platform probe IDs*, not gene symbols, so a
series matrix cannot be scored against a signature until it has been through
this module. The mapping lives in the platform's GPL annotation, which is
fetched once and cached like any other artefact.

Three decisions here are deliberate and worth defending:

* **Multi-target probes are dropped, not resolved.** GEO annotation encodes a
  probe matching several genes as ``BNIP3 /// BNIP3P1``. Silently taking the
  first target would attribute one measurement to one arbitrarily chosen gene.
  They are excluded by default and counted in the report; ``multi="first"``
  exists for reproducing legacy analyses that did take the first.
* **Unmapped probes are dropped and counted.** Control probes and retired
  identifiers have no symbol; carrying them forward inflates the matrix and
  the apparent gene coverage of every signature.
* **Collapsing many probes onto one symbol is a named rule**, reusing
  :func:`hypoxiapipe.harmonise.symbols.collapse_duplicate_rows`, because
  averaging probes with different hybridisation efficiency mixes scales.

Symbols produced here are still platform-era symbols. Run
:func:`hypoxiapipe.harmonise.symbols.harmonise_symbols` afterwards to map them
onto the pinned HGNC release.
"""

from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass, field

import pandas as pd

from hypoxiapipe.errors import ParseError
from hypoxiapipe.harmonise.symbols import collapse_duplicate_rows
from hypoxiapipe.ingest.cache import Cache

GPL_ANNOT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/{stub}/{acc}/annot/{acc}.annot.gz"

#: Candidate symbol columns, in preference order. GEO is not consistent about
#: which of these a platform carries, so the parser looks for the first present
#: rather than assuming one.
SYMBOL_COLUMNS = (
    "Gene symbol",
    "Gene Symbol",
    "GENE_SYMBOL",
    "gene_symbol",
    "Symbol",
    "SYMBOL",
    "ILMN_Gene",
)

MULTI_RULES = ("drop", "first")

_MULTI_SEP = re.compile(r"\s*///\s*")


def gpl_annot_url(platform: str) -> str:
    """Canonical GPL annotation URL for a platform such as 'GPL10558'."""
    if not re.fullmatch(r"GPL\d+", platform):
        raise ValueError(f"not a GEO platform accession: {platform!r}")
    stub = platform[:-3] + "nnn"
    return GPL_ANNOT_URL.format(stub=stub, acc=platform)


@dataclass(frozen=True)
class ProbeMap:
    """A parsed probe -> symbol mapping and how it was derived."""

    platform: str
    mapping: dict[str, str]
    symbol_column: str
    n_probes: int
    multi_target: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unmapped: tuple[str, ...] = ()
    multi_rule: str = "drop"

    @property
    def n_mapped(self) -> int:
        """Number of probes carrying exactly one usable symbol."""
        return len(self.mapping)

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary for the QC report and run manifest."""
        return {
            "platform": self.platform,
            "symbol_column": self.symbol_column,
            "n_probes": self.n_probes,
            "n_mapped": self.n_mapped,
            "n_multi_target": len(self.multi_target),
            "n_unmapped": len(self.unmapped),
            "multi_rule": self.multi_rule,
        }


@dataclass(frozen=True)
class ProbeMapReport:
    """What applying a :class:`ProbeMap` did to a matrix."""

    platform: str
    n_rows_in: int
    n_rows_out: int
    n_dropped_unmapped: int
    n_dropped_multi: int
    collapsed: dict[str, int] = field(default_factory=dict)
    collapse_rule: str = "max_mean"

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary for the QC report and run manifest."""
        return {
            "platform": self.platform,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_dropped_unmapped": self.n_dropped_unmapped,
            "n_dropped_multi": self.n_dropped_multi,
            "n_collapsed_symbols": len(self.collapsed),
            "collapse_rule": self.collapse_rule,
        }


def parse_gpl_annotation(
    text: str,
    platform: str | None = None,
    multi: str = "drop",
) -> ProbeMap:
    """Parse GPL annotation text into a probe -> symbol map.

    Handles both the ``.annot`` layout and the SOFT platform table: metadata
    lines prefixed with ``^``, ``!`` or ``#``, then a tab-delimited table
    between ``!platform_table_begin`` and ``!platform_table_end``. Files
    without the delimiters are treated as a bare table, which is what a
    hand-trimmed fixture usually is.
    """
    if multi not in MULTI_RULES:
        raise ValueError(f"unknown multi-target rule {multi!r} (choose from {MULTI_RULES})")

    lines = text.splitlines()
    if platform is None:
        for line in lines:
            if line.startswith("^PLATFORM"):
                platform = line.split("=", 1)[-1].strip()
                break
            if line.startswith("!Platform_geo_accession"):
                platform = line.split("=", 1)[-1].strip().strip('"')
                break
    platform = platform or "unknown"

    begin = next((i for i, ln in enumerate(lines) if ln.startswith("!platform_table_begin")), None)
    if begin is None:
        table_lines = [ln for ln in lines if ln.strip() and ln[0] not in "^!#"]
    else:
        end = next(
            (i for i, ln in enumerate(lines) if ln.startswith("!platform_table_end")), len(lines)
        )
        table_lines = [ln for ln in lines[begin + 1 : end] if ln.strip()]

    if not table_lines:
        raise ParseError(f"{platform}: no annotation table found")

    table = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", dtype=str, low_memory=False)
    if table.empty:
        raise ParseError(f"{platform}: annotation table is empty")

    id_col = table.columns[0]
    symbol_col = next((c for c in SYMBOL_COLUMNS if c in table.columns), None)
    if symbol_col is None:
        raise ParseError(
            f"{platform}: no gene-symbol column in annotation "
            f"(looked for {SYMBOL_COLUMNS}; found {list(table.columns)[:12]})"
        )

    mapping: dict[str, str] = {}
    multi_target: dict[str, tuple[str, ...]] = {}
    unmapped: list[str] = []

    for probe, raw in zip(table[id_col], table[symbol_col], strict=True):
        probe = str(probe).strip()
        if not probe or probe.lower() == "nan":
            continue
        value = "" if pd.isna(raw) else str(raw).strip()
        if not value or value.lower() in {"nan", "---", "null"}:
            unmapped.append(probe)
            continue
        targets = tuple(t for t in (s.strip() for s in _MULTI_SEP.split(value)) if t)
        if len(targets) > 1:
            multi_target[probe] = targets
            if multi == "first":
                mapping[probe] = targets[0]
            continue
        mapping[probe] = targets[0]

    return ProbeMap(
        platform=platform,
        mapping=mapping,
        symbol_column=symbol_col,
        n_probes=int(table.shape[0]),
        multi_target=multi_target,
        unmapped=tuple(unmapped),
        multi_rule=multi,
    )


def fetch_probe_map(
    platform: str,
    cache: Cache,
    multi: str = "drop",
) -> tuple[ProbeMap, str, str]:
    """Return ``(probe_map, url, checksum)``, downloading the GPL once if needed."""
    url = gpl_annot_url(platform)
    entry = cache.fetch(f"geo/{platform}.annot.gz", url)
    raw = entry.path.read_bytes()
    text = (
        gzip.decompress(raw).decode("utf-8", errors="replace")
        if raw[:2] == b"\x1f\x8b"
        else raw.decode("utf-8", errors="replace")
    )
    return parse_gpl_annotation(text, platform=platform, multi=multi), url, entry.checksum


def apply_probe_map(
    expr: pd.DataFrame,
    probe_map: ProbeMap,
    collapse_rule: str = "max_mean",
) -> tuple[pd.DataFrame, ProbeMapReport]:
    """Relabel a probe-indexed matrix with gene symbols and collapse duplicates.

    Rows whose probe has no single symbol under ``probe_map`` are dropped, and
    the counts are returned rather than logged and forgotten - a matrix that
    silently loses half its rows during mapping is the sort of thing that only
    surfaces later as unexplained low signature coverage.
    """
    index = [str(i).strip() for i in expr.index]
    keep_positions = [i for i, probe in enumerate(index) if probe in probe_map.mapping]
    n_multi = sum(
        1 for probe in index if probe in probe_map.multi_target and probe not in probe_map.mapping
    )
    n_unmapped = len(index) - len(keep_positions) - n_multi

    mapped = expr.iloc[keep_positions].copy()
    mapped.index = pd.Index(
        [probe_map.mapping[index[i]] for i in keep_positions], name="gene", dtype=object
    )
    if mapped.empty:
        raise ParseError(
            f"{probe_map.platform}: no probes in the matrix matched the annotation "
            "- check the matrix is from this platform"
        )

    mapped, collapsed = collapse_duplicate_rows(mapped, rule=collapse_rule)
    return mapped, ProbeMapReport(
        platform=probe_map.platform,
        n_rows_in=int(expr.shape[0]),
        n_rows_out=int(mapped.shape[0]),
        n_dropped_unmapped=int(n_unmapped),
        n_dropped_multi=int(n_multi),
        collapsed=collapsed,
        collapse_rule=collapse_rule,
    )
