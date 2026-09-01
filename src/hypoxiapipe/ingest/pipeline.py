"""Cohort build orchestration: accession in, QC'd harmonised cohort out.

This is the Phase 2 exit criterion in one function. The ordering of the stages
is not arbitrary, and two steps in particular have to happen where they do:

* **Symbol harmonisation precedes scoring, always.** A signature listing
  ``CYR61`` scores nothing against a matrix labelled ``CCN1``. The gene is
  present, the coverage report says it is missing, and the score is quietly
  computed over fewer genes than intended.
* **Restriction to the analysis set precedes any standardisation.** Per-gene
  z-scores are relative to the samples in the matrix, so scoring the full
  matrix and then subsetting to patients with outcome data is a different
  analysis from subsetting first. Building the cohort down to its analysis
  population here means every downstream score is computed over one declared,
  hashed population.

Nothing in this module standardises or scores. It hands the next phase a
matrix, a clinical table with a usable endpoint, a QC report, and the
provenance trail that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hypoxiapipe.errors import IngestError
from hypoxiapipe.harmonise.aliases import AliasTable, load_table
from hypoxiapipe.harmonise.probes import ProbeMapReport, apply_probe_map, fetch_probe_map
from hypoxiapipe.harmonise.symbols import SymbolReport, harmonise_symbols
from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.cohort import Cohort
from hypoxiapipe.ingest.endpoints import EndpointReport, derive_endpoint
from hypoxiapipe.ingest.geo import load_geo
from hypoxiapipe.ingest.spec import CohortSpec
from hypoxiapipe.ingest.tcga_build import load_tcga
from hypoxiapipe.qc.platform import infer_scale
from hypoxiapipe.qc.report import QCReport, run_qc
from hypoxiapipe.signatures.registry import Signature

TIME_COLUMN = "time_months"
EVENT_COLUMN = "event"


@dataclass
class BuildResult:
    """A built cohort and everything that describes how it was built."""

    cohort: Cohort
    qc: QCReport
    endpoint: EndpointReport | None = None
    symbols: SymbolReport | None = None
    probes: ProbeMapReport | None = None
    tcga: dict[str, Any] | None = None
    expectation_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable build summary for the Phase 3 run manifest."""
        return {
            "cohort": self.cohort.summary(),
            "provenance": self.cohort.provenance.to_dict(),
            "qc": self.qc.to_dict(),
            "endpoint": self.endpoint.to_dict() if self.endpoint else None,
            "symbols": self.symbols.to_dict() if self.symbols else None,
            "probes": self.probes.to_dict() if self.probes else None,
            "tcga": self.tcga,
            "expectation_failures": self.expectation_failures,
        }


def _load_local(spec: CohortSpec) -> Cohort:
    """Load a cohort from local expression/clinical tables (fixtures, exports)."""
    from hypoxiapipe.ingest.cohort import Provenance

    base = Path(spec.path or "")
    expr_path = base / "expr.tsv" if base.is_dir() else base
    clin_path = (base / "clinical.tsv") if base.is_dir() else base.with_name("clinical.tsv")
    if not expr_path.exists() or not clin_path.exists():
        raise IngestError(
            f"{spec.name}: expected expr.tsv and clinical.tsv under {base} "
            f"(looked for {expr_path} and {clin_path})"
        )
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    clinical = pd.read_csv(clin_path, sep="\t", index_col=0, dtype=str)
    prov = Provenance(
        source="local", accession=None, url=str(base), platform=spec.platform
    ).with_step("load_local", expr=str(expr_path), clinical=str(clin_path))
    return Cohort.align(
        name=spec.name, expr=expr, clinical=clinical, provenance=prov, min_samples=1
    )


def build_cohort(
    spec: CohortSpec,
    cache: Cache,
    signatures: list[Signature] | None = None,
    alias_table: AliasTable | None = None,
    strict_expectations: bool = True,
    min_samples: int = 30,
) -> BuildResult:
    """Build one cohort from its spec, harmonise it, and QC it.

    Parameters
    ----------
    spec : CohortSpec
        The cohort specification: accession, endpoint mapping, expectations.
    cache : Cache
        Content-addressed download cache; in offline mode a miss raises.
    signatures : list[Signature] | None
        Signatures whose gene coverage should be checked during QC.
    alias_table : AliasTable | None
        Pinned symbol authority; loaded from the spec's release if omitted.
    strict_expectations : bool
        Raise when the built cohort contradicts ``spec.expect``.
    min_samples : int
        Floor below which the cohort is not worth building.

    """
    # -- 1. acquire -------------------------------------------------------
    tcga_report: dict[str, Any] | None = None
    if spec.source == "geo":
        assert spec.accession is not None
        cohort = load_geo(spec.accession, cache, name=spec.name, min_samples=min_samples)
    elif spec.source == "local":
        cohort = _load_local(spec)
    elif spec.source == "tcga":
        assert spec.accession is not None
        cohort, tcga_report = load_tcga(spec, cache, min_samples=min_samples)
    else:  # pragma: no cover - guarded by CohortSpec validation
        raise IngestError(f"{spec.name}: unsupported source {spec.source!r}")

    # -- 2. probes -> symbols (array platforms only) ----------------------
    probe_report: ProbeMapReport | None = None
    platform = spec.platform or cohort.provenance.platform
    if spec.source == "geo" and platform:
        probe_map, url, checksum = fetch_probe_map(platform, cache, multi=spec.multi_probe_rule)
        expr, probe_report = apply_probe_map(
            cohort.expr, probe_map, collapse_rule=spec.collapse_rule
        )
        cohort = cohort.with_expression(
            expr,
            "map_probes_to_symbols",
            url=url,
            annotation_checksum=checksum,
            **probe_report.to_dict(),  # carries platform, counts and collapse rule
        )

    # -- 3. symbols -> pinned authority -----------------------------------
    if alias_table is not None:
        table = alias_table
    elif spec.symbol_authority:
        table = load_table(release=spec.symbol_authority)
    else:
        table = load_table()
    expr, symbol_report = harmonise_symbols(cohort.expr, table, collapse_rule=spec.collapse_rule)
    cohort = cohort.with_expression(expr, "harmonise_symbols", **symbol_report.to_dict())
    # Provenance is frozen: record the authority by replacement, not mutation.
    cohort = Cohort(
        name=cohort.name,
        expr=cohort.expr,
        clinical=cohort.clinical,
        provenance=replace(cohort.provenance, symbol_authority=table.authority),
    )

    # -- 4. scale ---------------------------------------------------------
    if spec.log2_transform is None:
        scale = infer_scale(cohort.expr)
        needs_log = scale.scale == "linear"
    else:
        needs_log = bool(spec.log2_transform)
    if needs_log:
        logged = pd.DataFrame(
            np.log2(cohort.expr.clip(lower=0).to_numpy(dtype=float) + 1.0),
            index=cohort.expr.index,
            columns=cohort.expr.columns,
        )
        cohort = cohort.with_expression(logged, "log2_transform", pseudocount=1.0)

    # -- 5. endpoint + analysis set ---------------------------------------
    endpoint_report: EndpointReport | None = None
    if spec.source == "tcga":
        # The TCGA loader derives the endpoint from the CDR or GDC clinical
        # table, because the encoding differs from the per-column mapping a
        # GEO spec declares. Restriction to the analysis set still happens
        # here, so every source narrows to its population the same way.
        cohort = cohort.restrict_to_analysis_set(TIME_COLUMN, EVENT_COLUMN)
    elif spec.endpoint is not None:
        clinical, endpoint_report = derive_endpoint(
            cohort.clinical, spec.endpoint, time_out=TIME_COLUMN, event_out=EVENT_COLUMN
        )
        cohort = Cohort(
            name=cohort.name,
            expr=cohort.expr,
            clinical=clinical,
            provenance=cohort.provenance.with_step("derive_endpoint", **endpoint_report.to_dict()),
        )
        cohort = cohort.restrict_to_analysis_set(TIME_COLUMN, EVENT_COLUMN)

    # -- 6. QC ------------------------------------------------------------
    has_endpoint = spec.endpoint is not None or spec.source == "tcga"
    qc = run_qc(
        cohort,
        signatures=signatures,
        min_samples=min_samples,
        event_col=EVENT_COLUMN if has_endpoint else None,
    )

    # -- 7. expectations --------------------------------------------------
    n_events = endpoint_report.n_events if endpoint_report else None
    if spec.source == "tcga":
        n_events = int(cohort.clinical[EVENT_COLUMN].sum())
    failures = spec.expect.check(spec.name, cohort.n_samples, cohort.n_genes, n_events)
    if failures and strict_expectations:
        raise IngestError(
            "cohort does not match its pinned expectations:\n  "
            + "\n  ".join(failures)
            + "\nEither the upstream data changed or the spec is wrong; "
            "confirm which before updating the spec."
        )

    return BuildResult(
        cohort=cohort,
        qc=qc,
        endpoint=endpoint_report,
        symbols=symbol_report,
        probes=probe_report,
        tcga=tcga_report if spec.source == "tcga" else None,
        expectation_failures=failures,
    )
