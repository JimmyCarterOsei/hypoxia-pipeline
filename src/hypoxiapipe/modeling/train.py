"""Model fitting and nested cross-validation.

Two estimators, both from scikit-survival: penalised Cox (``coxnet``) for a
sparse linear risk score, and a random survival forest for a non-linear
comparison. Neither is the point on its own. The point is the evaluation
protocol around them.

Why nested CV and not a single split
------------------------------------
Choosing a penalty by cross-validation and then reporting the cross-validated
score of the chosen penalty reports the best of several attempts as if it were
one. The inner loop selects hyperparameters; the outer loop scores the whole
selection procedure on data the inner loop never saw. The outer score is the
one that transfers.

Why preprocessing is fitted inside the loop
-------------------------------------------
Standardising before splitting lets held-out samples influence the features
used to train. It never errors and the optimism is small enough to look like a
real result. Every fold here fits its own :class:`ReferenceScaler` on the
training half only, and :func:`assert_no_leakage` exists so that the property
is tested rather than asserted in a comment.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.modeling.preprocessing import ReferenceScaler
from hypoxiapipe.provenance.hashing import hash_json

DEFAULT_ALPHAS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
ESTIMATORS = ("coxnet", "rsf")


class ModelingError(HypoxiapipeError):
    """A model could not be fitted or evaluated as requested."""


def to_structured(time: Any, event: Any) -> Any:
    """Convert time and event arrays into scikit-survival's structured format."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if time.shape != event.shape:
        raise ModelingError(f"time and event lengths differ ({time.shape} vs {event.shape})")
    if not np.isin(event, (0, 1)).all():
        raise ModelingError("event must be coded 0/1")
    if not np.all(np.isfinite(time)) or np.any(time <= 0):
        raise ModelingError("time must be finite and strictly positive")
    structured: Any = Surv.from_arrays(event=event.astype(bool), time=time)
    return structured


def make_estimator(kind: str, alpha: float | None = None, seed: int = 0) -> Any:
    """Construct an estimator by name."""
    if kind == "coxnet":
        return CoxnetSurvivalAnalysis(
            l1_ratio=0.9,
            alphas=[alpha] if alpha is not None else None,
            fit_baseline_model=False,
        )
    if kind == "rsf":
        return RandomSurvivalForest(
            n_estimators=200,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
    raise ModelingError(f"unknown estimator {kind!r} (choose from {ESTIMATORS})")


@contextmanager
def _quiet_penalty_search() -> Any:
    """Silence coxnet's all-zero-coefficient warning during a penalty search.

    A penalty that shrinks every coefficient to zero is a legitimate point on
    the grid and the inner loop simply scores it poorly; warning once per fold
    per alpha buries anything worth reading.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="all coefficients are zero")
        yield


def _risk_scores(model: Any, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(features), dtype=float).ravel()


def _c_index(structured: Any, risk: np.ndarray) -> float:
    return float(concordance_index_censored(structured["event"], structured["time"], risk)[0])


@dataclass(frozen=True)
class FoldResult:
    """One outer fold: what was chosen, and how it scored on unseen data."""

    fold: int
    estimator: str
    alpha: float | None
    n_train: int
    n_test: int
    n_test_events: int
    c_index: float
    n_features_selected: int | None = None
    scaler_checksum: str = ""


@dataclass
class CVResult:
    """The outcome of a nested cross-validation."""

    estimator: str
    folds: list[FoldResult] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def c_index_mean(self) -> float:
        """Mean outer-fold concordance."""
        return float(np.mean([f.c_index for f in self.folds])) if self.folds else float("nan")

    @property
    def c_index_sd(self) -> float:
        """Standard deviation across outer folds - the honest spread."""
        if len(self.folds) < 2:
            return float("nan")
        return float(np.std([f.c_index for f in self.folds], ddof=1))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable summary for reports and manifests."""
        return {
            "estimator": self.estimator,
            "n_folds": len(self.folds),
            "c_index_mean": self.c_index_mean,
            "c_index_sd": self.c_index_sd,
            "c_index_per_fold": [f.c_index for f in self.folds],
            "alphas_selected": [f.alpha for f in self.folds],
            "params": self.params,
            "fingerprint": hash_json(
                {
                    "estimator": self.estimator,
                    "params": self.params,
                    "c_index": [round(f.c_index, 8) for f in self.folds],
                }
            ),
        }

    def to_markdown(self) -> str:
        """Render the fold-by-fold table and the summary line."""
        lines = [
            f"# Nested CV - {self.estimator}",
            "",
            "| fold | n train | n test | events | alpha | C-index |",
            "|---|---|---|---|---|---|",
        ]
        for f in self.folds:
            alpha = "-" if f.alpha is None else f"{f.alpha:.4g}"
            lines.append(
                f"| {f.fold} | {f.n_train} | {f.n_test} | {f.n_test_events} | "
                f"{alpha} | {f.c_index:.3f} |"
            )
        lines += [
            "",
            f"**Outer C-index: {self.c_index_mean:.3f} (SD {self.c_index_sd:.3f})** "
            f"across {len(self.folds)} folds",
        ]
        return "\n".join(lines)


def _select_alpha(
    features: pd.DataFrame,
    structured: Any,
    alphas: tuple[float, ...],
    inner_splits: int,
    seed: int,
) -> float:
    """Choose a penalty by inner-loop CV, refitting the scaler in every inner fold."""
    inner = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    scores: dict[float, list[float]] = {a: [] for a in alphas}

    for train_idx, val_idx in inner.split(features.T):
        train = features.iloc[:, train_idx]
        val = features.iloc[:, val_idx]
        scaler = ReferenceScaler().fit(train)
        x_train = scaler.transform(train).T.to_numpy(dtype=float)
        x_val = scaler.transform(val).T.to_numpy(dtype=float)
        y_train, y_val = structured[train_idx], structured[val_idx]
        if y_val["event"].sum() < 2:
            continue
        for alpha in alphas:
            try:
                with _quiet_penalty_search():
                    model = make_estimator("coxnet", alpha=alpha, seed=seed).fit(x_train, y_train)
                scores[alpha].append(_c_index(y_val, _risk_scores(model, x_val)))
            except (ValueError, ArithmeticError):
                # An unfittable penalty is a point the grid rejects, not a
                # reason to abort the search.
                continue

    mean_scores = {a: float(np.mean(v)) for a, v in scores.items() if v}
    if not mean_scores:
        raise ModelingError("no penalty could be evaluated on the inner folds")
    return max(mean_scores, key=lambda a: mean_scores[a])


def nested_cv(
    features: pd.DataFrame,
    time: Any,
    event: Any,
    estimator: str = "coxnet",
    outer_splits: int = 5,
    inner_splits: int = 3,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    seed: int = 0,
) -> CVResult:
    """Run nested cross-validation with per-fold frozen preprocessing.

    Parameters
    ----------
    features : pd.DataFrame
        Genes x samples, matching the pipeline's orientation throughout.
    time : array-like
        Follow-up time, strictly positive.
    event : array-like
        Event indicator coded 0/1.
    estimator : str
        ``coxnet`` or ``rsf``.
    outer_splits : int
        Outer folds; each scores the whole selection procedure on unseen data.
    inner_splits : int
        Inner folds used only to choose the penalty (``coxnet`` only).
    alphas : tuple[float, ...]
        Penalty grid searched by the inner loop.
    seed : int
        Seed for fold assignment and the forest.

    """
    if estimator not in ESTIMATORS:
        raise ModelingError(f"unknown estimator {estimator!r} (choose from {ESTIMATORS})")
    structured = to_structured(time, event)
    if features.shape[1] != len(structured):
        raise ModelingError(
            f"features has {features.shape[1]} samples but {len(structured)} outcomes; "
            "the matrix must be genes x samples with columns aligned to the endpoint"
        )
    n_events = int(structured["event"].sum())
    if n_events < outer_splits * 2:
        raise ModelingError(
            f"{n_events} events across {outer_splits} outer folds is too few to fit; "
            "reduce the folds or report a single fitted model with its caveats instead"
        )

    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    result = CVResult(
        estimator=estimator,
        params={
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "alphas": list(alphas) if estimator == "coxnet" else None,
            "seed": seed,
            "n_genes": int(features.shape[0]),
            "n_samples": int(features.shape[1]),
            "n_events": n_events,
        },
    )

    for fold, (train_idx, test_idx) in enumerate(outer.split(features.T), start=1):
        train = features.iloc[:, train_idx]
        test = features.iloc[:, test_idx]
        y_train, y_test = structured[train_idx], structured[test_idx]

        # Fitted on the training half only, then frozen for the test half.
        scaler = ReferenceScaler().fit(train)
        x_train = scaler.transform(train).T.to_numpy(dtype=float)
        x_test = scaler.transform(test).T.to_numpy(dtype=float)

        alpha = None
        if estimator == "coxnet":
            alpha = _select_alpha(train, y_train, alphas, inner_splits, seed)
        model = make_estimator(estimator, alpha=alpha, seed=seed).fit(x_train, y_train)

        selected = None
        if estimator == "coxnet":
            selected = int(np.count_nonzero(np.asarray(model.coef_).ravel()))

        result.folds.append(
            FoldResult(
                fold=fold,
                estimator=estimator,
                alpha=alpha,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_test_events=int(y_test["event"].sum()),
                c_index=_c_index(y_test, _risk_scores(model, x_test)),
                n_features_selected=selected,
                scaler_checksum=scaler.checksum,
            )
        )
    return result


@dataclass(frozen=True)
class FittedModel:
    """A model plus the frozen scaler it must be applied with."""

    estimator: str
    model: Any
    scaler: ReferenceScaler
    alpha: float | None
    n_train: int
    train_population_hash: str | None = None

    @property
    def n_selected(self) -> int | None:
        """Number of non-zero coefficients, for sparse linear models."""
        coef = getattr(self.model, "coef_", None)
        return None if coef is None else int(np.count_nonzero(np.asarray(coef).ravel()))

    def predict(self, matrix: pd.DataFrame, strict: bool = True) -> pd.Series:
        """Score an external cohort using the training-time statistics.

        The scaler travels with the model precisely so that this cannot be done
        with the new cohort's own means and standard deviations - which would
        make the prediction depend on who else happened to be measured.
        """
        scaled = self.scaler.transform(matrix, strict=strict)
        risk = _risk_scores(self.model, scaled.T.to_numpy(dtype=float))
        return pd.Series(risk, index=matrix.columns, name=f"{self.estimator}_risk")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable description for the run manifest."""
        return {
            "estimator": self.estimator,
            "alpha": self.alpha,
            "n_train": self.n_train,
            "n_features_in": len(self.scaler.genes),
            "n_features_selected": self.n_selected,
            "scaler_checksum": self.scaler.checksum,
            "train_population_hash": self.train_population_hash,
        }


def fit_final(
    features: pd.DataFrame,
    time: Any,
    event: Any,
    estimator: str = "coxnet",
    alpha: float | None = None,
    inner_splits: int = 3,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    seed: int = 0,
    population_hash: str | None = None,
) -> FittedModel:
    """Fit the deployable model on all training data, with its scaler frozen.

    Report the nested-CV estimate as the model's expected performance; this
    model's fit to its own training data is not an estimate of anything.
    """
    structured = to_structured(time, event)
    if estimator == "coxnet" and alpha is None:
        alpha = _select_alpha(features, structured, alphas, inner_splits, seed)

    scaler = ReferenceScaler().fit(features, population_hash=population_hash)
    x = scaler.transform(features).T.to_numpy(dtype=float)
    with _quiet_penalty_search():
        model = make_estimator(estimator, alpha=alpha, seed=seed).fit(x, structured)
    return FittedModel(
        estimator=estimator,
        model=model,
        scaler=scaler,
        alpha=alpha,
        n_train=int(features.shape[1]),
        train_population_hash=population_hash,
    )


def assert_no_leakage(features: pd.DataFrame, train_idx: Any, test_idx: Any) -> None:
    """Raise if a scaler fitted on training samples reflects the test samples.

    Used by the test suite to check the property directly rather than trusting
    that the fold loop was written correctly: standardising the training half
    must give the same constants whatever the held-out half contains.
    """
    train = features.iloc[:, list(train_idx)]
    baseline = ReferenceScaler().fit(train).checksum

    perturbed = features.copy()
    test_columns = perturbed.columns[list(test_idx)]
    perturbed[test_columns] = perturbed[test_columns] + 1000.0
    after = ReferenceScaler().fit(perturbed.iloc[:, list(train_idx)]).checksum

    if baseline != after:
        raise ModelingError(
            "leakage: the training scaler changed when only held-out samples changed"
        )
