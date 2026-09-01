"""Cohort QC: structural checks, signature coverage, and a reportable result.

Findings carry a severity. ``FAIL`` means the cohort should not proceed;
``WARN`` means proceed but state it in the paper's methods; ``INFO`` is
description. ``QCReport.raise_on_fail()`` turns the first category into a
non-zero exit, so a broken cohort stops the pipeline rather than producing a
plausible-looking hazard ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from hypoxiapipe.errors import QCFailedError
from hypoxiapipe.ingest.cohort import Cohort
from hypoxiapipe.qc.platform import ScaleReport, infer_scale
from hypoxiapipe.signatures.registry import Signature


class Level(StrEnum):
    """Severity of a QC finding."""

    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Finding:
    """One QC observation."""

    level: Level
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QCReport:
    """Collected findings for one cohort."""

    cohort: str
    findings: list[Finding] = field(default_factory=list)
    scale: ScaleReport | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, level: Level, code: str, message: str, **detail: Any) -> None:
        """Record a finding."""
        self.findings.append(Finding(level=level, code=code, message=message, detail=detail))

    @property
    def failures(self) -> list[Finding]:
        """Return all FAIL-level findings."""
        return [f for f in self.findings if f.level is Level.FAIL]

    @property
    def warnings(self) -> list[Finding]:
        """Return all WARN-level findings."""
        return [f for f in self.findings if f.level is Level.WARN]

    @property
    def ok(self) -> bool:
        """Return True if nothing failed."""
        return not self.failures

    def raise_on_fail(self) -> None:
        """Raise ``QCFailedError`` if any check failed."""
        if self.failures:
            lines = "\n".join(f"  [{f.code}] {f.message}" for f in self.failures)
            raise QCFailedError(f"{self.cohort}: {len(self.failures)} QC failure(s)\n{lines}")

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a dictionary."""
        return {
            "cohort": self.cohort,
            "ok": self.ok,
            "summary": self.summary,
            "scale": self.scale.to_dict() if self.scale else None,
            "findings": [
                {"level": f.level.value, "code": f.code, "message": f.message, **f.detail}
                for f in self.findings
            ],
        }

    def to_json(self) -> str:
        """Return the report as indented JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        """Return the report as markdown, for the HTML report in Phase 3."""
        icon = {Level.INFO: "-", Level.WARN: "!", Level.FAIL: "x"}
        head = [f"# QC report - {self.cohort}", ""]
        for key, value in self.summary.items():
            head.append(f"- **{key}**: {value}")
        if self.scale:
            head += [
                "",
                f"- **scale**: {self.scale.scale} ({self.scale.assay}), "
                f"range {self.scale.minimum:.2f} to {self.scale.maximum:.2f}"
                + (" (sampled)" if self.scale.sampled else ""),
            ]
        head += ["", "## Findings", ""]
        for f in self.findings:
            head.append(f"{icon[f.level]} `{f.code}` **{f.level.value}** - {f.message}")
        head += [
            "",
            f"**Result: {'PASS' if self.ok else 'FAIL'}** "
            f"({len(self.failures)} failures, {len(self.warnings)} warnings)",
        ]
        return "\n".join(head)


def run_qc(
    cohort: Cohort,
    signatures: list[Signature] | None = None,
    max_gene_na: float = 0.20,
    max_sample_na: float = 0.20,
    min_coverage: float = 0.90,
    min_samples: int = 30,
    min_events: int = 10,
    event_col: str | None = None,
) -> QCReport:
    """Run structural and coverage QC over a cohort."""
    rep = QCReport(cohort=cohort.name)
    expr = cohort.expr
    rep.summary = {
        "n_genes": cohort.n_genes,
        "n_samples": cohort.n_samples,
        "population_hash": cohort.population_hash,
        "expr_checksum": cohort.expr_checksum,
        "platform": cohort.provenance.platform,
    }

    if cohort.n_samples < min_samples:
        rep.add(
            Level.FAIL,
            "n_samples",
            f"only {cohort.n_samples} samples (minimum {min_samples})",
            n_samples=cohort.n_samples,
        )

    rep.scale = infer_scale(expr)
    if rep.scale.recommendation:
        rep.add(
            Level.WARN,
            "scale",
            rep.scale.recommendation,
            scale=rep.scale.scale,
            assay=rep.scale.assay,
            max=rep.scale.maximum,
        )

    gene_na = expr.isna().mean(axis=1)
    bad_genes = gene_na[gene_na > max_gene_na]
    if len(bad_genes):
        rep.add(
            Level.WARN,
            "gene_missingness",
            f"{len(bad_genes)} genes exceed {max_gene_na:.0%} missing values",
            n_genes=int(len(bad_genes)),
            worst=sorted(bad_genes.index[:10].astype(str)),
        )

    sample_na = expr.isna().mean(axis=0)
    bad_samples = sample_na[sample_na > max_sample_na]
    if len(bad_samples):
        rep.add(
            Level.FAIL,
            "sample_missingness",
            f"{len(bad_samples)} samples exceed {max_sample_na:.0%} missing values",
            samples=sorted(bad_samples.index[:10].astype(str)),
        )

    variance = expr.var(axis=1, skipna=True)
    zero_var = variance[(variance.fillna(0) == 0)]
    if len(zero_var):
        rep.add(
            Level.WARN,
            "zero_variance",
            f"{len(zero_var)} genes have zero variance and cannot be z-scored",
            n_genes=int(len(zero_var)),
            examples=sorted(zero_var.index[:10].astype(str)),
        )

    if expr.index.duplicated().any():
        n = int(expr.index.duplicated().sum())
        rep.add(
            Level.FAIL,
            "duplicate_genes",
            f"{n} duplicate gene symbols in the matrix; harmonise before scoring",
            n_duplicates=n,
        )

    constant_samples = expr.std(axis=0, skipna=True)
    flat = constant_samples[constant_samples.fillna(0) == 0]
    if len(flat):
        rep.add(
            Level.FAIL,
            "flat_samples",
            f"{len(flat)} samples have zero variance across genes",
            samples=sorted(flat.index[:10].astype(str)),
        )

    if event_col:
        if event_col not in cohort.clinical.columns:
            rep.add(Level.FAIL, "endpoint_missing", f"no clinical column '{event_col}'")
        else:
            events = pd.to_numeric(cohort.clinical[event_col], errors="coerce")
            n_events = int(np.nansum(events.to_numpy(dtype=float)))
            rep.summary["n_events"] = n_events
            level = Level.WARN if n_events >= min_events else Level.FAIL
            if n_events < min_events * 2:
                rep.add(
                    level,
                    "event_count",
                    f"{n_events} events; with fewer than ~{min_events * 2} the "
                    "quartile analysis is underpowered and per-SD Cox should lead",
                    n_events=n_events,
                )

    for sig in signatures or []:
        found = [g for g in sig.genes if g in expr.index]
        coverage = len(found) / sig.n_genes
        missing = [g for g in sig.genes if g not in expr.index]
        level = Level.INFO if coverage >= min_coverage else Level.WARN
        rep.add(
            level,
            f"coverage:{sig.name}",
            f"{sig.name}: {len(found)}/{sig.n_genes} genes present ({coverage:.0%})",
            coverage=round(coverage, 3),
            missing=missing[:20],
            checksum=sig.checksum,
        )

    return rep
