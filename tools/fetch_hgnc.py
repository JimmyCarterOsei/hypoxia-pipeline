#!/usr/bin/env python3
"""Build a full HGNC alias table from the official complete set.

The table bundled with the package is a curated subset covering this project's
signatures - deliberately small, so tests are deterministic and offline. For a
production run, build the complete mapping with this script and pin its hash.

Usage
-----
    python tools/fetch_hgnc.py --out data/hgnc_aliases_2026-08-01.tsv
    hypoxiapipe cohort harmonise expr.tsv --table data/hgnc_aliases_2026-08-01.tsv --out h.tsv

The source file is the HGNC "complete set" TSV published at genenames.org. Its
URL and layout change occasionally, so it is a parameter rather than a constant.

Pinning
-------
The printed checksum goes into ``BUNDLED_CHECKSUMS`` in
``hypoxiapipe/harmonise/aliases.py`` *by hand*. That is intentional friction: a
changed symbol mapping changes every downstream score, so it should be a commit
someone reviews, not a silent refresh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

DEFAULT_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)

# Columns in the HGNC complete set that carry retired symbols.
ALIAS_COLUMNS = {"prev_symbol": "previous", "alias_symbol": "alias"}


def build_table(text: str) -> list[tuple[str, str, str, str]]:
    """Turn the HGNC complete set into (alias, approved, kind, note) rows."""
    lines = text.splitlines()
    header = lines[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    for required in ("symbol", "status"):
        if required not in idx:
            raise SystemExit(f"unexpected HGNC layout: no '{required}' column")

    rows: list[tuple[str, str, str, str]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        if parts[idx["status"]] != "Approved":
            continue
        approved = parts[idx["symbol"]].strip()
        for column, kind in ALIAS_COLUMNS.items():
            if column not in idx:
                continue
            raw = parts[idx[column]].strip().strip('"')
            for alias in filter(None, (a.strip() for a in raw.split("|"))):
                if alias.upper() != approved.upper():
                    rows.append((alias.upper(), approved, kind, ""))
    return rows


def main() -> int:
    """Download, convert, write, and print the checksum to pin."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output TSV path.")
    ap.add_argument("--url", default=DEFAULT_URL, help="HGNC complete-set URL.")
    ap.add_argument("--source", type=Path, help="Use a local complete-set file instead.")
    args = ap.parse_args()

    if args.source:
        text = args.source.read_text()
        origin = str(args.source)
    else:
        with urlopen(args.url, timeout=300) as resp:  # noqa: S310 - explicit URL argument
            text = resp.read().decode("utf-8", errors="replace")
        origin = args.url

    rows = build_table(text)
    ambiguous = {a for a, _, _, _ in rows if sum(1 for r in rows if r[0] == a) > 1}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write(f"# HGNC complete set, converted by tools/fetch_hgnc.py\n# source: {origin}\n")
        fh.write(f"# rows: {len(rows)}; aliases with multiple approved targets: {len(ambiguous)}\n")
        fh.write("alias\tapproved\tkind\tnote\n")
        for alias, approved, kind, note in sorted(rows):
            fh.write(f"{alias}\t{approved}\t{kind}\t{note}\n")

    from hypoxiapipe.harmonise.aliases import checksum_text

    checksum = checksum_text(args.out.read_text())
    print(f"wrote {args.out} ({len(rows)} rows, {len(ambiguous)} ambiguous aliases)")
    print(f"checksum: {checksum}")
    print("Pin this in BUNDLED_CHECKSUMS only after reviewing the diff.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
