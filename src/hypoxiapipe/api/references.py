"""Registered scoring references.

A reference is the thing that makes scoring a single sample meaningful.

Per-gene z-scoring is cohort-relative: a gene's score depends on the mean and
standard deviation across the samples in the matrix. Send one patient and there
is no distribution to standardise against, so the score is undefined - not
imprecise, undefined. Send five patients and the score exists but means
"relative to these four others", which is not a clinically interpretable
quantity either.

A reference solves this by freezing the statistics from a named cohort. A new
sample is then scored against *that* population rather than against whoever
happens to be in the same batch, and the answer is reproducible and
attributable: every response names the reference, its checksum, and the
population hash of the cohort it was derived from.

This is the same discipline as ADR 0007, one layer out. There, preprocessing
was frozen so a model could be applied to an external cohort. Here it is frozen
so a model can be applied to one patient.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.modeling.preprocessing import ReferenceScaler
from hypoxiapipe.signatures.registry import Signature, load_bundled

#: Directory holding registered references, one JSON file each.
REFERENCE_DIR_ENV = "HYPOXIAPIPE_REFERENCES"
DEFAULT_REFERENCE_DIR = Path("references")


class ReferenceError(HypoxiapipeError):
    """A scoring reference is missing, malformed, or does not fit the request."""


def reference_dir() -> Path:
    """Return the directory holding registered references."""
    return Path(os.environ.get(REFERENCE_DIR_ENV) or DEFAULT_REFERENCE_DIR)


@dataclass(frozen=True)
class ScoringReference:
    """Frozen per-gene statistics from one named cohort, for one signature."""

    reference_id: str
    signature: str
    signature_checksum: str
    method: str
    scaler: ReferenceScaler
    cohort: str
    population_hash: str | None
    n_train: int
    created: str

    @property
    def genes(self) -> tuple[str, ...]:
        """Genes the reference carries statistics for."""
        return self.scaler.genes

    def describe(self) -> dict[str, Any]:
        """Return the provenance a scored response should carry."""
        return {
            "reference_id": self.reference_id,
            "signature": self.signature,
            "signature_checksum": self.signature_checksum,
            "method": self.method,
            "scaler_checksum": self.scaler.checksum,
            "cohort": self.cohort,
            "population_hash": self.population_hash,
            "n_train": self.n_train,
            "n_genes": len(self.genes),
            "created": self.created,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the serialisable form written to disk."""
        return {**self.describe(), "scaler": self.scaler.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScoringReference:
        """Rebuild a reference, verifying the scaler's checksum."""
        try:
            scaler = ReferenceScaler.from_dict(raw["scaler"])
        except KeyError as exc:
            raise ReferenceError(f"reference is missing '{exc.args[0]}'") from exc
        return cls(
            reference_id=str(raw["reference_id"]),
            signature=str(raw["signature"]),
            signature_checksum=str(raw.get("signature_checksum", "")),
            method=str(raw.get("method", "rowmean")),
            scaler=scaler,
            cohort=str(raw.get("cohort", "unknown")),
            population_hash=raw.get("population_hash"),
            n_train=int(raw.get("n_train", 0)),
            created=str(raw.get("created", "")),
        )

    def score(self, matrix: pd.DataFrame, strict: bool = True) -> pd.Series:
        """Score samples against the frozen reference statistics.

        Unlike cohort-relative scoring this is well defined for a single
        sample, because the distribution comes from the reference rather than
        from the samples supplied.
        """
        standardised = self.scaler.transform(matrix, strict=strict)
        return standardised.mean(axis=0)


def _signature_for(reference: ScoringReference) -> Signature:
    return load_bundled(reference.signature)


def build_reference(
    reference_id: str,
    cohort_expr: pd.DataFrame,
    signature: Signature,
    cohort_name: str,
    population_hash: str | None = None,
    method: str = "rowmean",
    created: str | None = None,
) -> ScoringReference:
    """Freeze per-gene statistics from a cohort for later single-sample scoring."""
    from datetime import UTC, datetime  # noqa: PLC0415

    present = [g for g in signature.genes if g in cohort_expr.index]
    if not present:
        raise ReferenceError(
            f"none of {signature.name}'s {signature.n_genes} genes are in the cohort"
        )
    scaler = ReferenceScaler().fit(cohort_expr.loc[present], population_hash=population_hash)
    return ScoringReference(
        reference_id=reference_id,
        signature=signature.name,
        signature_checksum=signature.checksum,
        method=method,
        scaler=scaler,
        cohort=cohort_name,
        population_hash=population_hash,
        n_train=int(cohort_expr.shape[1]),
        created=created or datetime.now(UTC).isoformat(timespec="seconds"),
    )


def save_reference(reference: ScoringReference, directory: str | Path | None = None) -> Path:
    """Write a reference into the reference directory."""
    out = Path(directory) if directory else reference_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{reference.reference_id}.json"
    path.write_text(json.dumps(reference.to_dict(), indent=2))
    return path


def load_reference(reference_id: str, directory: str | Path | None = None) -> ScoringReference:
    """Load one registered reference by id."""
    src = Path(directory) if directory else reference_dir()
    path = src / f"{reference_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in src.glob("*.json")) if src.exists() else []
        raise ReferenceError(f"no reference '{reference_id}' in {src} (available: {available})")
    return ScoringReference.from_dict(json.loads(path.read_text()))


def list_references(directory: str | Path | None = None) -> list[ScoringReference]:
    """Load every registered reference, skipping none silently."""
    src = Path(directory) if directory else reference_dir()
    if not src.exists():
        return []
    out: list[ScoringReference] = []
    for path in sorted(src.glob("*.json")):
        out.append(ScoringReference.from_dict(json.loads(path.read_text())))
    return out
