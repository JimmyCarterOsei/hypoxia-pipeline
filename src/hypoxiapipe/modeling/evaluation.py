"""Paired summaries of repeated held-out risks, without cross-fold risk comparisons.

Patient-cluster bootstrap weights are shared across models, folds and repeats.
They describe uncertainty conditional on the saved fitted outer predictions;
this is not a bootstrap that refits model selection and training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hypoxiapipe.modeling.train import ModelingError


def paired_bootstrap(
    predictions: dict[str, pd.DataFrame], draws: int = 2000, seed: int = 902099
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarise within-fold C and paired patient-bootstrap percentile intervals.

    No risks from different fitted models are ranked against one another.
    Bootstrap patients once per draw, retaining each patient's multiplicity
    across all its held-out appearances. Same-time event/censor pairs follow
    Harrell/scikit-survival conventions; risk differences <=1e-8 are ties.
    """
    names = list(predictions)
    if not names or draws < 2:
        raise ModelingError("need models and at least two bootstrap draws")
    tables = {
        name: frame.sort_values(["repeat", "fold", "patient"]).reset_index(drop=True)
        for name, frame in predictions.items()
    }
    reference = tables[names[0]]
    if reference.duplicated(["repeat", "patient"]).any():
        raise ModelingError("each patient must have one held-out prediction per repeat")
    ids = sorted(reference.patient.unique())
    if not all(set(group.patient) == set(ids) for _, group in reference.groupby("repeat")):
        raise ModelingError("every repeat must cover the same patients")
    for frame in tables.values():
        if not frame[["repeat", "fold", "patient", "time", "event"]].equals(
            reference[["repeat", "fold", "patient", "time", "event"]]
        ):
            raise ModelingError("paired comparisons require identical held-out patients/outcomes")
        if not np.isfinite(frame.risk).all():
            raise ModelingError("nonfinite outer risks")
    pos = {p: i for i, p in enumerate(ids)}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    fold_values: list[np.ndarray] = []
    fold_boots: list[np.ndarray] = []
    denominators: list[int] = []
    rows: list[dict[str, Any]] = []
    for (repeat, fold), group in reference.groupby(["repeat", "fold"], sort=True):
        time = group.time.to_numpy()
        event = group.event.to_numpy(dtype=bool)
        # i had an event; j remains at risk after i, or is censored at i's time.
        comparable = event[:, None] & (
            (time[:, None] < time[None, :]) | ((time[:, None] == time[None, :]) & ~event[None, :])
        )
        i, j = np.nonzero(comparable)
        if not len(i):
            raise ModelingError("outer fold has no comparable pairs")
        concordance = []
        for name in names:
            risk = tables[name].loc[group.index, "risk"].to_numpy()
            difference = risk[i] - risk[j]
            concordance.append(
                (difference > 1e-8).astype(float) + 0.5 * (np.abs(difference) <= 1e-8)
            )
        credit = np.array(concordance)
        indices = np.array([pos[p] for p in group.patient])
        pair_weights = weights[:, indices[i]] * weights[:, indices[j]]
        den = pair_weights.sum(axis=1)
        boot = np.full((draws, len(names)), np.nan)
        valid = den > 0
        boot[valid] = (pair_weights[valid] @ credit.T) / den[valid, None]
        values = credit.mean(axis=1)
        fold_values.append(values)
        fold_boots.append(boot)
        denominators.append(len(i))
        rows.extend(
            {
                "model": name,
                "repeat": repeat,
                "fold": fold,
                "c_index": float(values[k]),
                "comparable_pairs": len(i),
            }
            for k, name in enumerate(names)
        )
    values = np.array(fold_values)
    boots = np.mean(np.array(fold_boots), axis=0)
    boots = boots[np.isfinite(boots).all(axis=1)]
    if len(boots) < draws * 0.8:
        raise ModelingError("too many bootstrap draws lack comparable pairs")
    summaries, pairs = [], []
    for k, name in enumerate(names):
        low, high = np.quantile(boots[:, k], [0.025, 0.975])
        summaries.append(
            {
                "model": name,
                "n_outer_fits": len(values),
                "mean_c": float(values[:, k].mean()),
                "sd_c": float(values[:, k].std(ddof=1)),
                "median_c": float(np.median(values[:, k])),
                "min_c": float(values[:, k].min()),
                "max_c": float(values[:, k].max()),
                "pair_weighted_c": float(np.average(values[:, k], weights=denominators)),
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_valid": len(boots),
            }
        )
        for j in range(k):
            low, high = np.quantile(boots[:, k] - boots[:, j], [0.025, 0.975])
            difference = values[:, k] - values[:, j]
            pairs.append(
                {
                    "comparison": f"{name} minus {names[j]}",
                    "mean_difference": difference.mean(),
                    "sd_difference": difference.std(ddof=1),
                    "median_difference": np.median(difference),
                    "ci_low": low,
                    "ci_high": high,
                    "folds_positive": int((difference > 0).sum()),
                    "n_outer_fits": len(values),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(pairs), pd.DataFrame(rows)
