"""Pinned gene-symbol authority.

Symbol mapping is treated exactly like a signature: a specific, checksummed
release rather than "whatever HGNC says today". If the table changes, the hash
changes, and every downstream manifest shows which mapping produced the result.

The bundled table is a **curated subset** covering the symbols that matter to
this project's signatures. That is stated in the file header and in
``AliasTable.is_subset`` rather than glossed over - a partial table presented as
complete would be the same class of error the registry exists to prevent. Use
``tools/fetch_hgnc.py`` to build the full table and re-pin the checksum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from hypoxiapipe.errors import AliasTableError

_DATA_PACKAGE = "hypoxiapipe.harmonise.data"
DEFAULT_RELEASE = "2024-07-01"

# Pinned hash of the bundled curated table. Recomputed deliberately, never
# auto-updated: a changed mapping table is a change to every result.
BUNDLED_CHECKSUMS = {
    "2024-07-01": "sha256:0753e62261a1069a7f14df404cfe330b56df73d0c9334bbf7bc931c167aae435",
}


@dataclass(frozen=True)
class AliasTable:
    """An alias-to-approved-symbol mapping from a specific release."""

    release: str
    mapping: dict[str, tuple[str, ...]]
    checksum: str
    is_subset: bool
    n_entries: int

    @property
    def authority(self) -> str:
        """Return a human-readable authority string, e.g. 'HGNC 2024-07-01'."""
        return f"HGNC {self.release}" + (" (curated subset)" if self.is_subset else "")

    def approved_for(self, symbol: str) -> tuple[str, ...]:
        """Return the approved symbol(s) an alias maps to; empty if unknown."""
        return self.mapping.get(symbol.strip().upper(), ())


def _parse_table(text: str) -> tuple[dict[str, tuple[str, ...]], bool]:
    mapping: dict[str, list[str]] = {}
    is_subset = "curated subset" in text.lower() or "NOT the full HGNC" in text
    seen_header = False
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        if not seen_header and parts[0].lower() == "alias":
            seen_header = True
            continue
        if len(parts) < 2:
            raise AliasTableError(f"malformed alias row: {line!r}")
        alias, approved = parts[0].upper(), parts[1]
        mapping.setdefault(alias, [])
        if approved not in mapping[alias]:
            mapping[alias].append(approved)
    if not mapping:
        raise AliasTableError("alias table is empty")
    return {k: tuple(v) for k, v in mapping.items()}, is_subset


def checksum_text(text: str) -> str:
    """SHA-256 of the alias table text, ignoring comment lines and blank lines."""
    body = "\n".join(
        line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    return f"sha256:{hashlib.sha256(body.encode()).hexdigest()}"


def load_table(
    path: str | Path | None = None, release: str = DEFAULT_RELEASE, verify: bool = True
) -> AliasTable:
    """Load an alias table, verifying its checksum against the pinned value.

    Passing an explicit `path` (e.g. a full HGNC table built by
    ``tools/fetch_hgnc.py``) skips pin verification but records the actual hash,
    so the manifest still identifies the mapping exactly.
    """
    if path is not None:
        text = Path(path).read_text()
        mapping, is_subset = _parse_table(text)
        return AliasTable(
            release=f"file:{Path(path).name}",
            mapping=mapping,
            checksum=checksum_text(text),
            is_subset=is_subset,
            n_entries=len(mapping),
        )

    # Located by iterating the data package rather than by joinpath().is_file().
    # Under an editable install `resources.files()` can return a MultiplexedPath
    # whose joinpath().is_file() reports False for a file that iterdir() plainly
    # lists - which produced the memorable error "no bundled alias table for
    # release '2024-07-01' (installed: ['2024-07-01'])". Iteration is the form
    # that works in both layouts.
    wanted = f"hgnc_aliases_{release}.tsv"
    ref = next((f for f in resources.files(_DATA_PACKAGE).iterdir() if f.name == wanted), None)
    if ref is None or not ref.is_file():
        # List the files that exist, not the releases a constant says should:
        # an installation missing its data reported the absent release as
        # available, which is the least useful thing it could have said.
        present = sorted(
            f.name.removeprefix("hgnc_aliases_").removesuffix(".tsv")
            for f in resources.files(_DATA_PACKAGE).iterdir()
            if f.name.startswith("hgnc_aliases_") and f.name.endswith(".tsv")
        )
        declared = sorted(BUNDLED_CHECKSUMS)
        if not present:
            raise AliasTableError(
                f"no alias table files are installed (declared releases: {declared}). "
                "The package data is missing - check the installation, and that the "
                "tables are tracked in version control rather than ignored."
            )
        raise AliasTableError(
            f"no bundled alias table for release {release!r} (installed: {present})"
        )
    text = ref.read_text()
    actual = checksum_text(text)
    if verify:
        expected = BUNDLED_CHECKSUMS.get(release)
        if expected and expected != actual:
            raise AliasTableError(
                f"alias table checksum mismatch for release {release}.\n"
                f"  pinned: {expected}\n  actual: {actual}\n"
                "The symbol mapping changed. That changes results, so re-pin it "
                "deliberately in BUNDLED_CHECKSUMS rather than in passing."
            )
    mapping, is_subset = _parse_table(text)
    return AliasTable(
        release=release,
        mapping=mapping,
        checksum=actual,
        is_subset=is_subset,
        n_entries=len(mapping),
    )
