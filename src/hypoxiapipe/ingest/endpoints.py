"""Survival endpoint derivation.

Public cohorts encode the same endpoint in incompatible ways: time in months
here and days there, event as ``0/1`` in one and ``"Yes"/"No"`` or
``"BCR"/"NED"`` in another. Every downstream number depends on getting this
right, so the mapping is declared per cohort in its spec and applied here,
rather than being written inline four times with four sets of assumptions.

Two things this module makes explicit:

* **Unit conversion is declared, never guessed.** A cohort recording follow-up
  in days looks identical to one recording months until the hazard ratios come
  out wrong. ``time_unit`` is required in the spec.
* **Administrative censoring is a recorded transformation.** Capping follow-up
  at five years means a patient who recurs at 70 months becomes an event-free
  observation at 60 - a real change to the data, so it appends a provenance
  step and reports how many observations it altered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hypoxiapipe.errors import EndpointError

TIME_UNITS = {"days": 1.0 / 30.4375, "weeks": 7.0 / 30.4375, "months": 1.0, "years": 12.0}

#: Values treated as an event when the spec does not name them explicitly.
DEFAULT_EVENT_VALUES = ("1", "yes", "y", "true", "event", "recurrence", "bcr", "dead", "deceased")
DEFAULT_CENSOR_VALUES = ("0", "no", "n", "false", "censored", "ned", "alive", "no recurrence")


@dataclass(frozen=True)
class EndpointSpec:
    """How to build a survival endpoint from one cohort's clinical columns."""

    name: str
    time_column: str
    event_column: str
    time_unit: str = "months"
    event_values: tuple[str, ...] = ()
    censor_values: tuple[str, ...] = ()
    cap_months: float | None = None

    def __post_init__(self) -> None:
        """Validate the declared time unit."""
        if self.time_unit not in TIME_UNITS:
            raise EndpointError(
                f"endpoint '{self.name}': unknown time_unit {self.time_unit!r} "
                f"(choose from {sorted(TIME_UNITS)})"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, object], name: str = "endpoint") -> EndpointSpec:
        """Build from a YAML mapping, requiring the fields that cannot be guessed."""
        for required in ("time_column", "event_column", "time_unit"):
            if not raw.get(required):
                raise EndpointError(
                    f"endpoint '{name}': '{required}' is required and has no safe default"
                )

        def _tokens(key: str) -> tuple[str, ...]:
            values = raw.get(key) or ()
            if isinstance(values, str):  # a bare scalar in YAML is a common slip
                values = [values]
            if not isinstance(values, list | tuple):
                raise EndpointError(f"endpoint '{name}': '{key}' must be a list of values")
            return tuple(str(v).lower() for v in values)

        cap = raw.get("cap_months")
        if cap is not None and not isinstance(cap, int | float | str):
            raise EndpointError(f"endpoint '{name}': 'cap_months' must be a number")
        return cls(
            name=str(raw.get("name", name)),
            time_column=str(raw["time_column"]),
            event_column=str(raw["event_column"]),
            time_unit=str(raw["time_unit"]),
            event_values=_tokens("event_values"),
            censor_values=_tokens("censor_values"),
            cap_months=float(cap) if cap is not None else None,
        )


@dataclass(frozen=True)
class EndpointReport:
    """What endpoint derivation produced, and what it had to discard."""

    name: str
    time_column: str
    event_column: str
    n_input: int
    n_usable: int
    n_events: int
    n_dropped_missing: int
    n_unparsed_event: int
    n_censored_by_cap: int = 0
    cap_months: float | None = None
    unrecognised_values: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_dropped(self) -> int:
        """Samples lost because time or event could not be resolved."""
        return self.n_dropped_missing + self.n_unparsed_event

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary for the QC report and run manifest."""
        return {
            "endpoint": self.name,
            "time_column": self.time_column,
            "event_column": self.event_column,
            "n_input": self.n_input,
            "n_usable": self.n_usable,
            "n_events": self.n_events,
            "n_dropped_missing": self.n_dropped_missing,
            "n_unparsed_event": self.n_unparsed_event,
            "n_censored_by_cap": self.n_censored_by_cap,
            "cap_months": self.cap_months,
            "unrecognised_values": list(self.unrecognised_values),
        }


def _coerce_time(values: pd.Series, unit: str) -> pd.Series:
    """Return follow-up time in months, non-numeric entries as NaN."""
    numeric = pd.to_numeric(
        values.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False), errors="coerce"
    )
    return numeric.astype(float) * TIME_UNITS[unit]


def _coerce_event(values: pd.Series, spec: EndpointSpec) -> tuple[pd.Series, list[str]]:
    """Return a 0/1 event series and any values that could not be interpreted."""
    event_set = set(spec.event_values) or set(DEFAULT_EVENT_VALUES)
    censor_set = set(spec.censor_values) or set(DEFAULT_CENSOR_VALUES)

    out: list[float] = []
    unrecognised: list[str] = []
    for raw in values:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            out.append(float("nan"))
            continue
        token = str(raw).strip().lower()
        if token in event_set:
            out.append(1.0)
        elif token in censor_set:
            out.append(0.0)
        else:
            out.append(float("nan"))
            unrecognised.append(str(raw))
    return pd.Series(out, index=values.index, dtype=float), sorted(set(unrecognised))


def derive_endpoint(
    clinical: pd.DataFrame,
    spec: EndpointSpec,
    time_out: str = "time_months",
    event_out: str = "event",
) -> tuple[pd.DataFrame, EndpointReport]:
    """Add numeric ``time_months`` and ``event`` columns to a clinical table.

    Samples whose time or event cannot be resolved are **retained** with NaN;
    dropping them is a separate, recorded decision made by
    :meth:`Cohort.restrict_to_analysis_set`, so that the count of samples lost
    to a missing endpoint stays visible instead of vanishing silently here.
    """
    for column in (spec.time_column, spec.event_column):
        if column not in clinical.columns:
            raise EndpointError(
                f"endpoint '{spec.name}': column {column!r} not in clinical table "
                f"(available: {sorted(clinical.columns)[:15]})"
            )

    out = clinical.copy()
    time = _coerce_time(clinical[spec.time_column], spec.time_unit)
    event, unrecognised = _coerce_event(clinical[spec.event_column], spec)

    # An event value that was present but not interpretable is a different
    # problem from one that was simply absent, and is worth reporting apart:
    # it usually means the spec's event_values are wrong for this cohort.
    unparsed_mask = event.isna() & clinical[spec.event_column].notna()
    n_unparsed_event = int(unparsed_mask.sum())

    n_capped = 0
    if spec.cap_months is not None:
        cap = float(spec.cap_months)
        beyond = (time > cap) & time.notna()
        n_capped = int((beyond & (event == 1)).sum())
        event = event.where(~beyond, 0.0)
        time = time.clip(upper=cap)

    usable = time.notna() & event.notna()
    out[time_out] = time
    out[event_out] = event

    report = EndpointReport(
        name=spec.name,
        time_column=spec.time_column,
        event_column=spec.event_column,
        n_input=int(clinical.shape[0]),
        n_usable=int(usable.sum()),
        n_events=int(np.nansum(event[usable].to_numpy())),
        n_dropped_missing=int((~usable & ~unparsed_mask).sum()),
        n_unparsed_event=n_unparsed_event,
        n_censored_by_cap=n_capped,
        cap_months=spec.cap_months,
        unrecognised_values=tuple(unrecognised[:10]),
    )
    return out, report
