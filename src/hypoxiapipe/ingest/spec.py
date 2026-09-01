"""Cohort specifications.

A cohort spec is to a dataset what a signature spec is to a gene list: a
declared, versioned description that the code refuses to proceed without. It
names the accession, the platform, how the survival endpoint is encoded, and
what the cohort is *expected* to contain.

The expectations are the point. ``expect.n_samples`` is not documentation - it
is asserted at build time, so a silently re-versioned GDC release or an
amended GEO submission fails the build instead of shifting every hazard ratio
by a little. That is the same failure mode as the mislabelled gene vector, one
layer down: the input changed and nothing said so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from hypoxiapipe.errors import IngestError
from hypoxiapipe.ingest.endpoints import EndpointSpec

_SPEC_PACKAGE = "hypoxiapipe.ingest.specs"
SOURCES = ("geo", "tcga", "local")


@dataclass(frozen=True)
class Expectation:
    """What a correctly built cohort should look like."""

    n_samples: int | None = None
    n_samples_tolerance: int = 0
    min_genes: int | None = None
    min_events: int | None = None

    def check(self, name: str, n_samples: int, n_genes: int, n_events: int | None) -> list[str]:
        """Return a list of human-readable expectation violations."""
        problems: list[str] = []
        if self.n_samples is not None:
            delta = abs(n_samples - self.n_samples)
            if delta > self.n_samples_tolerance:
                problems.append(
                    f"{name}: expected {self.n_samples} samples "
                    f"(+/-{self.n_samples_tolerance}), built {n_samples}"
                )
        if self.min_genes is not None and n_genes < self.min_genes:
            problems.append(f"{name}: expected at least {self.min_genes} genes, built {n_genes}")
        if self.min_events is not None and n_events is not None and n_events < self.min_events:
            problems.append(f"{name}: expected at least {self.min_events} events, got {n_events}")
        return problems


@dataclass(frozen=True)
class CohortSpec:
    """Declarative description of one cohort."""

    name: str
    source: str
    accession: str | None = None
    platform: str | None = None
    endpoint: EndpointSpec | None = None
    expect: Expectation = field(default_factory=Expectation)
    symbol_authority: str | None = None
    collapse_rule: str = "max_mean"
    multi_probe_rule: str = "drop"
    log2_transform: bool | None = None
    path: str | None = None
    notes: str = ""

    # -- TCGA-specific ----------------------------------------------------
    workflow: str = "STAR - Counts"
    star_value_column: str = "tpm_unstranded"
    duplicate_aliquot_rule: str = "first"
    primary_tumours_only: bool = True
    clinical_source: str = "gdc"
    clinical_path: str | None = None
    cdr_endpoint: str = "PFI"
    tolerate_file_failures: int = 0

    def __post_init__(self) -> None:
        """Validate source-specific requirements."""
        if self.source not in SOURCES:
            raise IngestError(
                f"{self.name}: unknown source {self.source!r} (choose from {SOURCES})"
            )
        if self.source in {"geo", "tcga"} and not self.accession:
            raise IngestError(f"{self.name}: source '{self.source}' requires an accession")
        if self.source == "local" and not self.path:
            raise IngestError(f"{self.name}: source 'local' requires a path")
        if self.source == "tcga":
            if self.clinical_source not in {"gdc", "cdr"}:
                raise IngestError(
                    f"{self.name}: clinical_source must be 'gdc' or 'cdr', "
                    f"not {self.clinical_source!r}"
                )
            if self.clinical_source == "cdr" and not self.clinical_path:
                raise IngestError(
                    f"{self.name}: clinical_source 'cdr' requires clinical_path pointing at "
                    "your copy of the TCGA-CDR table (it is not redistributed)"
                )


def _parse(raw: dict[str, Any], origin: str) -> CohortSpec:
    name = raw.get("name")
    if not name:
        raise IngestError(f"{origin}: cohort spec has no 'name'")
    source = raw.get("source")
    if not source:
        raise IngestError(f"{origin}: cohort '{name}' has no 'source'")

    endpoint_raw = raw.get("endpoint")
    endpoint = (
        EndpointSpec.from_dict(endpoint_raw, name=f"{name}:endpoint")
        if isinstance(endpoint_raw, dict)
        else None
    )
    expect_raw = raw.get("expect") or {}
    expect = Expectation(
        n_samples=expect_raw.get("n_samples"),
        n_samples_tolerance=int(expect_raw.get("n_samples_tolerance", 0)),
        min_genes=expect_raw.get("min_genes"),
        min_events=expect_raw.get("min_events"),
    )
    return CohortSpec(
        name=str(name),
        source=str(source).lower(),
        accession=raw.get("accession"),
        platform=raw.get("platform"),
        endpoint=endpoint,
        expect=expect,
        symbol_authority=raw.get("symbol_authority"),
        collapse_rule=str(raw.get("collapse_rule", "max_mean")),
        multi_probe_rule=str(raw.get("multi_probe_rule", "drop")),
        log2_transform=raw.get("log2_transform"),
        path=raw.get("path"),
        notes=str(raw.get("notes", "")),
        workflow=str(raw.get("workflow", "STAR - Counts")),
        star_value_column=str(raw.get("star_value_column", "tpm_unstranded")),
        duplicate_aliquot_rule=str(raw.get("duplicate_aliquot_rule", "first")),
        primary_tumours_only=bool(raw.get("primary_tumours_only", True)),
        clinical_source=str(raw.get("clinical_source", "gdc")),
        clinical_path=raw.get("clinical_path"),
        cdr_endpoint=str(raw.get("cdr_endpoint", "PFI")),
        tolerate_file_failures=int(raw.get("tolerate_file_failures", 0)),
    )


def load_cohort_spec(path: str | Path) -> CohortSpec:
    """Load a cohort spec from a YAML file."""
    p = Path(path)
    return _parse(yaml.safe_load(p.read_text()), origin=str(p))


def _bundled_names() -> list[str]:
    files = resources.files(_SPEC_PACKAGE)
    return sorted(f.name.removesuffix(".yaml") for f in files.iterdir() if f.name.endswith(".yaml"))


def load_bundled_cohort(name: str) -> CohortSpec:
    """Load a bundled cohort spec by short name (e.g. 'cambridge')."""
    ref = resources.files(_SPEC_PACKAGE).joinpath(f"{name}.yaml")
    if not ref.is_file():
        raise IngestError(f"no bundled cohort '{name}' (available: {', '.join(_bundled_names())})")
    return _parse(yaml.safe_load(ref.read_text()), origin=f"bundled:{name}")


def list_bundled_cohorts() -> dict[str, CohortSpec | IngestError]:
    """Load every bundled cohort spec, returning errors rather than raising."""
    out: dict[str, CohortSpec | IngestError] = {}
    for name in _bundled_names():
        try:
            out[name] = load_bundled_cohort(name)
        except IngestError as exc:
            out[name] = exc
    return out
