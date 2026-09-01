"""Sample selection from clinical annotation.

A GEO series is whatever the submitters deposited, which is rarely the cohort a
prognostic analysis wants. GSE70768 carries 199 samples: primary tumours, the
benign tissue they were compared against, and a handful of castrate-resistant
disease. Scoring all of them together is not a larger version of the intended
analysis — it is a different one. Benign tissue widens every gene's apparent
dynamic range, which shifts the cohort-relative z-scores of the tumours; CRPC
is a different disease state whose recurrence endpoint means something else.

The TCGA path already handles this, because sample type is encoded in the
barcode and :func:`restrict_to_primary_tumours` reads it. GEO has no such
convention: the information is in a free-text characteristic whose column name
and values differ per submission. So it is declared per cohort in the spec
rather than guessed.

Filters run *before* endpoint derivation and before the analysis set is fixed,
so the population every score is relative to is the intended one. Each filter
records what it dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypoxiapipe.errors import IngestError
from hypoxiapipe.ingest.cohort import Cohort


@dataclass(frozen=True)
class SampleFilter:
    """One rule for which samples belong in a cohort.

    Exactly one of ``keep``, ``drop`` or ``drop_contains`` applies. Matching is
    case-insensitive on the string form of the value, because GEO free text is
    inconsistent about capitalisation.
    """

    column: str
    keep: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    drop_contains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require exactly one mode."""
        modes = [bool(self.keep), bool(self.drop), bool(self.drop_contains)]
        if sum(modes) != 1:
            raise IngestError(
                f"sample filter on '{self.column}' must set exactly one of "
                "keep, drop or drop_contains"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SampleFilter:
        """Build from a YAML mapping."""
        column = raw.get("column")
        if not column:
            raise IngestError("sample filter has no 'column'")

        def tokens(key: str) -> tuple[str, ...]:
            values = raw.get(key) or ()
            if isinstance(values, str):
                values = [values]
            return tuple(str(v) for v in values)

        return cls(
            column=str(column),
            keep=tokens("keep"),
            drop=tokens("drop"),
            drop_contains=tokens("drop_contains"),
        )

    def describe(self) -> str:
        """Return a one-line description for the provenance record."""
        if self.keep:
            return f"{self.column} in {list(self.keep)}"
        if self.drop:
            return f"{self.column} not in {list(self.drop)}"
        return f"{self.column} does not contain {list(self.drop_contains)}"


def apply_filters(
    cohort: Cohort, filters: list[SampleFilter]
) -> tuple[Cohort, list[dict[str, Any]]]:
    """Apply sample filters in order, returning the cohort and what each dropped."""
    reports: list[dict[str, Any]] = []
    current = cohort

    for rule in filters:
        if rule.column not in current.clinical.columns:
            raise IngestError(
                f"{current.name}: sample filter names column '{rule.column}', which is "
                f"not in the clinical table (available: {sorted(current.clinical.columns)[:12]}). "
                "Run 'hypoxiapipe cohort inspect' to see the columns GEO actually returns."
            )

        values = current.clinical[rule.column].astype(str).str.strip().str.lower()
        if rule.keep:
            wanted = {v.lower() for v in rule.keep}
            mask = values.isin(wanted)
        elif rule.drop:
            unwanted = {v.lower() for v in rule.drop}
            mask = ~values.isin(unwanted)
        else:
            patterns = [v.lower() for v in rule.drop_contains]
            mask = ~values.apply(lambda v, p=patterns: any(token in v for token in p))

        keep = [s for s, ok in zip(current.clinical.index, mask, strict=True) if ok]
        before = current.n_samples
        if not keep:
            raise IngestError(
                f"{current.name}: filter '{rule.describe()}' removed every sample. "
                f"Observed values: {sorted(set(values))[:10]}"
            )

        current = current.subset_samples(list(keep), reason=rule.describe())
        reports.append(
            {
                "filter": rule.describe(),
                "column": rule.column,
                "n_before": before,
                "n_after": current.n_samples,
                "n_dropped": before - current.n_samples,
            }
        )

    return current, reports
