"""Content-addressed signature registry.

Signatures are YAML specs carrying their source (DOI, table), gene list,
optional coefficients, default scoring method, and a SHA-256 checksum of
the canonicalised gene content. Loading a signature verifies the checksum;
any silent edit to the gene list fails loudly.

Motivation: a benchmark in the upstream research project was invalidated
by a comparator gene list that shared only 4 of 32 genes with the published
signature it was labelled as. The bug survived for months because nothing
verified gene-list provenance. This module makes that class of error
structurally impossible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from hypoxiapipe.errors import (
    ChecksumMismatchError,
    IncompleteSpecError,
    SignatureError,
)

_SPEC_PACKAGE = "hypoxiapipe.signatures.specs"

VALID_SCORING = frozenset({"rowmean", "median_z", "weighted"})


@dataclass(frozen=True)
class Signature:
    """A verified, immutable signature specification.

    `scoring` is required and must state the method the signature was
    PUBLISHED with. There is deliberately no default: silently assuming a
    method is the same class of error as silently assuming a gene list.
    Method-sensitivity is a separate, explicit analysis.
    """

    name: str
    genes: tuple[str, ...]
    scoring: str
    source: dict[str, Any] = field(default_factory=dict)
    coefficients: dict[str, float] | None = None
    checksum: str = ""
    notes: str = ""
    symbol_authority: str = ""  # e.g. "HGNC 2024-07-01" - pinned mapping release

    @property
    def n_genes(self) -> int:
        """Number of genes in the signature."""
        return len(self.genes)

    @property
    def weighted(self) -> bool:
        """True if the signature carries per-gene coefficients."""
        return self.coefficients is not None


def canonical_content(genes: list[str], coefficients: dict[str, float] | None) -> str:
    """Canonical string representation of signature content for hashing.

    Genes are sorted (order-independent); coefficients, if present, are
    rendered with fixed precision so the hash is stable across YAML
    round-trips and float formatting differences.
    """
    lines = sorted(g.strip() for g in genes)
    if coefficients is not None:
        lines += [f"{g}={coefficients[g]:.9f}" for g in sorted(coefficients)]
    return "\n".join(lines)


def compute_checksum(genes: list[str], coefficients: dict[str, float] | None = None) -> str:
    """SHA-256 of the canonical signature content, prefixed for clarity."""
    digest = hashlib.sha256(canonical_content(genes, coefficients).encode()).hexdigest()
    return f"sha256:{digest}"


def _parse_spec(raw: dict[str, Any], origin: str) -> Signature:
    name = raw.get("name")
    if not name:
        raise SignatureError(f"{origin}: spec has no 'name'")

    genes = raw.get("genes") or []
    if not genes:
        raise IncompleteSpecError(
            f"{origin}: signature '{name}' has an empty gene list. "
            "Fill it from the verified source and re-hash "
            "(hypoxiapipe sig hash <spec>)."
        )
    genes = [str(g).strip() for g in genes]
    if len(set(genes)) != len(genes):
        dupes = sorted({g for g in genes if genes.count(g) > 1})
        raise SignatureError(f"{origin}: duplicate genes in '{name}': {dupes}")

    coefficients = raw.get("coefficients")
    if coefficients is not None:
        coefficients = {str(k): float(v) for k, v in coefficients.items()}
        if set(coefficients) != set(genes):
            missing = sorted(set(genes) ^ set(coefficients))
            raise SignatureError(
                f"{origin}: '{name}' coefficients do not cover exactly the "
                f"gene list (symmetric difference: {missing})"
            )

    scoring = raw.get("scoring")
    if not scoring:
        raise IncompleteSpecError(
            f"{origin}: '{name}' has no 'scoring' method. State the method the "
            "signature was published with (rowmean | median_z | weighted); "
            "there is no default."
        )
    if scoring not in VALID_SCORING:
        raise SignatureError(
            f"{origin}: '{name}' has unknown scoring '{scoring}' (valid: {sorted(VALID_SCORING)})"
        )
    if scoring == "weighted" and coefficients is None:
        raise SignatureError(
            f"{origin}: '{name}' declares weighted scoring but carries no coefficients"
        )

    recorded = raw.get("checksum")
    if not recorded:
        raise IncompleteSpecError(
            f"{origin}: '{name}' has no checksum. Compute one with "
            "'hypoxiapipe sig hash' after verifying the gene list against source."
        )
    actual = compute_checksum(genes, coefficients)
    if recorded != actual:
        raise ChecksumMismatchError(
            f"{origin}: checksum mismatch for '{name}'.\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            "The gene content has changed since it was last verified against "
            "source. Re-verify against the cited table, then re-hash."
        )

    return Signature(
        name=name,
        genes=tuple(genes),
        scoring=str(scoring),
        source=raw.get("source", {}),
        coefficients=coefficients,
        checksum=recorded,
        notes=str(raw.get("notes", "")),
        symbol_authority=str(raw.get("symbol_authority", "")),
    )


def load_spec(path: str | Path) -> Signature:
    """Load and verify a signature spec from a YAML file path."""
    p = Path(path)
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    return _parse_spec(raw, origin=str(p))


def _iter_bundled_spec_names() -> list[str]:
    files = resources.files(_SPEC_PACKAGE)
    return sorted(f.name.removesuffix(".yaml") for f in files.iterdir() if f.name.endswith(".yaml"))


def load_bundled(name: str) -> Signature:
    """Load a spec bundled with the package by short name (e.g. 'smith20')."""
    ref = resources.files(_SPEC_PACKAGE).joinpath(f"{name}.yaml")
    if not ref.is_file():
        available = ", ".join(_iter_bundled_spec_names())
        raise SignatureError(f"no bundled signature '{name}' (available: {available})")
    raw = yaml.safe_load(ref.read_text())
    return _parse_spec(raw, origin=f"bundled:{name}")


def list_bundled(strict: bool = False) -> dict[str, Signature | SignatureError]:
    """Load every bundled spec.

    With strict=False (default), incomplete/invalid specs are returned as
    their exception rather than raised, so callers can report status.
    """
    out: dict[str, Signature | SignatureError] = {}
    for name in _iter_bundled_spec_names():
        try:
            out[name] = load_bundled(name)
        except SignatureError as exc:
            if strict:
                raise
            out[name] = exc
    return out
