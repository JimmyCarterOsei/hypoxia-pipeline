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
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from hypoxiapipe.errors import HypoxiapipeError
from hypoxiapipe.modeling.preprocessing import ReferenceScaler
from hypoxiapipe.modeling.rbridge import FrozenCox
from hypoxiapipe.provenance.hashing import hash_json
from hypoxiapipe.signatures.registry import load_bundled

DEFAULT_ALPHAS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
ESTIMATORS = ("coxnet", "rsf", "smith20_cox")


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


def make_estimator(kind: str, alpha: float | None = None, seed: int = 0, **params: Any) -> Any:
    """Construct a molecular estimator without fitting any preprocessing."""
    if kind == "smith20_cox":
        return FrozenCox()
    if kind == "coxnet":
        return CoxnetSurvivalAnalysis(
            l1_ratio=params.get("l1_ratio", 0.9),
            alphas=[alpha] if alpha is not None else None,
            fit_baseline_model=False,
            normalize=False,
            max_iter=100000,
            tol=1e-7,
        )
    if kind == "rsf":
        return RandomSurvivalForest(
            n_estimators=params.get("n_estimators", 200),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
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
    repeat: int = 1
    selected_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CVResult:
    """The outcome of a nested cross-validation."""

    estimator: str
    folds: list[FoldResult] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    splits: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    coefficients: list[dict[str, Any]] = field(default_factory=list)
    training: list[dict[str, Any]] = field(default_factory=list)
    tuning: list[dict[str, Any]] = field(default_factory=list)
    importance: list[dict[str, Any]] = field(default_factory=list)

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


def molecular_data(
    features: pd.DataFrame, time: Any, event: Any, genes: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, Any]:
    """Enforce feature names and align labeled outcomes before canonical sorting.

    Positional arrays are accepted for the legacy API; labeled Series must have
    exactly the feature sample IDs and are explicitly reindexed. Extra genes
    are ignored only when an explicit feature contract is supplied.
    """
    if not features.columns.is_unique or not features.index.is_unique:
        raise ModelingError("duplicate gene or sample labels")
    if not all(isinstance(v, str) for v in features.columns):
        raise ModelingError("sample IDs must be strings")
    contract = genes if genes is not None else tuple(str(g) for g in features.index)
    if not contract or len(set(contract)) != len(contract):
        raise ModelingError("feature contract must contain unique gene names")
    missing = [g for g in contract if g not in features.index]
    if missing:
        raise ModelingError(f"missing required genes: {missing}")
    if features.shape[1] != len(time) or features.shape[1] != len(event):
        raise ModelingError("genes x samples matrix and outcomes have different lengths")
    aligned = []
    for values in (time, event):
        if isinstance(values, pd.Series):
            if not values.index.is_unique or set(values.index) != set(features.columns):
                raise ModelingError("outcome sample IDs do not match expression")
            values = values.reindex(features.columns).to_numpy()
        aligned.append(np.asarray(values))
    structured = to_structured(*aligned)
    order = np.argsort(features.columns.to_numpy())
    matrix = features.loc[list(contract)].iloc[:, order].astype(float)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ModelingError("required expression values must be finite; no silent imputation")
    return matrix, structured[order]


def plan_cv(
    samples: list[str],
    event: Any,
    outer_splits: int = 5,
    inner_splits: int = 5,
    repeats: int = 5,
    seed: int = 902001,
) -> list[dict[str, Any]]:
    """Plan stratified nested folds using only canonical IDs and event indicators."""
    if samples != sorted(samples) or len(set(samples)) != len(samples):
        raise ModelingError("split planning requires unique sorted sample IDs")
    event = np.asarray(event, dtype=int)
    if len(event) != len(samples) or not np.isin(event, (0, 1)).all():
        raise ModelingError("invalid split event indicators")
    if min(outer_splits, inner_splits) < 2 or repeats < 1:
        raise ModelingError("need at least two folds and one repeat")
    if min(np.bincount(event, minlength=2)) < outer_splits * 2:
        raise ModelingError("too few to fit stratified outer folds: events or censored patients")
    ids = np.asarray(samples)
    plans = []
    for repeat in range(repeats):
        outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed + repeat)
        for fold, (tr, te) in enumerate(outer.split(ids, event), 1):
            inner_seed = seed + 1000 + 100 * repeat + fold
            model_seed = seed + 10000 + 100 * repeat + fold
            if min(np.bincount(event[tr], minlength=2)) < inner_splits:
                raise ModelingError("too few training outcomes for inner stratification")
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=inner_seed)
            plans.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "outer_seed": seed + repeat,
                    "inner_seed": inner_seed,
                    "model_seed": model_seed,
                    "train": ids[tr].tolist(),
                    "test": ids[te].tolist(),
                    "inner": [
                        {"train": ids[tr][a].tolist(), "validation": ids[tr][b].tolist()}
                        for a, b in inner.split(tr, event[tr])
                    ],
                }
            )
    return plans


def fit_partition(
    features: pd.DataFrame,
    structured: Any,
    estimator: str,
    params: dict[str, Any],
    seed: int,
) -> FittedModel:
    """Fit one already-aligned training partition; never receive test samples."""
    if estimator == "smith20_cox" and tuple(features.index) != tuple(load_bundled("smith20").genes):
        raise ModelingError("fixed Smith20 requires its exact ordered 20-gene contract")
    scaler = ReferenceScaler().fit(features)
    if scaler.detail["n_zero_variance"]:
        raise ModelingError("zero-variance training gene in the fixed feature contract")
    x = scaler.transform(features).T.to_numpy(dtype=float)
    composite_mean, composite_sd = 0.0, 1.0
    if estimator == "smith20_cox":
        raw = x.mean(axis=1)
        composite_mean, composite_sd = float(raw.mean()), float(raw.std(ddof=1))
        if not np.isfinite(composite_sd) or composite_sd <= 1e-8:
            raise ModelingError("degenerate fixed-signature composite")
        x = ((raw - composite_mean) / composite_sd)[:, None]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arguments = dict(params)
        alpha = arguments.pop("alpha", None)
        model = make_estimator(estimator, alpha=alpha, seed=seed, **arguments).fit(x, structured)
    messages = tuple(str(w.message) for w in caught)
    serious = [m for m in messages if "all coefficients are zero" not in m]
    if serious:
        raise ModelingError(f"fit warnings: {serious}")
    if not np.isfinite(_risk_scores(model, x)).all():
        raise ModelingError("nonfinite training predictions")
    return FittedModel(
        estimator,
        model,
        scaler,
        alpha,
        features.shape[1],
        composite_mean=composite_mean,
        composite_sd=composite_sd,
        selected_params=dict(params),
        warnings=messages,
    )


def select_parameters(
    features: pd.DataFrame,
    structured: Any,
    estimator: str,
    candidates: list[dict[str, Any]],
    inner: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune on training-only inner folds, rejecting incomplete candidate evaluations.

    Exact ties retain the first prespecified candidate. An all-zero CoxNet is
    a valid constant-risk candidate; convergence warnings are not silently used.
    """
    if not candidates:
        raise ModelingError("empty parameter search")
    if estimator == "smith20_cox":
        return {}, []
    pos = {s: i for i, s in enumerate(features.columns)}
    evaluations: list[dict[str, Any]] = []
    for number, params in enumerate(candidates):
        scores: list[float] = []
        failures: list[str] = []
        messages: list[str] = []
        for fold, split in enumerate(inner, 1):
            tr, va = split["train"], split["validation"]
            if set(tr) & set(va) or set(tr) | set(va) != set(pos):
                raise ModelingError("inner split crosses its training population")
            try:
                fitted = fit_partition(
                    features[tr], structured[[pos[i] for i in tr]], estimator, params, seed + fold
                )
                scores.append(
                    _c_index(
                        structured[[pos[i] for i in va]], fitted.predict(features[va]).to_numpy()
                    )
                )
                messages.extend(fitted.warnings)
            except (ValueError, ArithmeticError, ModelingError) as exc:
                failures.append(f"inner fold {fold}: {exc}")
        evaluations.append(
            {
                "candidate": number,
                "params": params,
                "inner_c": scores,
                "valid": not failures and len(scores) == len(inner),
                "mean_c": float(np.mean(scores)) if not failures and scores else None,
                "failures": failures,
                "warnings": messages,
            }
        )
    valid = [e for e in evaluations if e["valid"]]
    if not valid:
        raise ModelingError(f"no candidate completed every inner fold: {evaluations}")
    winner = max(valid, key=lambda row: row["mean_c"])
    return dict(winner["params"]), evaluations


def evaluate_fold(
    features: pd.DataFrame,
    structured: Any,
    estimator: str,
    split: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[FittedModel, np.ndarray, list[dict[str, Any]]]:
    """Tune and fit on outer training IDs, then apply frozen preprocessing to test IDs."""
    pos = {s: i for i, s in enumerate(features.columns)}
    tr, te = split["train"], split["test"]
    if set(tr) & set(te) or set(tr) | set(te) != set(pos):
        raise ModelingError("outer split does not partition the population")
    for group in [tr, te]:
        if len(group) != len(set(group)):
            raise ModelingError("duplicate IDs in outer split")
    for inner in split["inner"]:
        a, b = inner["train"], inner["validation"]
        if set(a) & set(b) or set(a) | set(b) != set(tr):
            raise ModelingError("inner split crosses its training population")
        if len(a) != len(set(a)) or len(b) != len(set(b)):
            raise ModelingError("duplicate IDs in inner split")
    train, y_train = features[tr], structured[[pos[i] for i in tr]]
    params, tuning = select_parameters(
        train, y_train, estimator, candidates, split["inner"], split["model_seed"]
    )
    fitted = fit_partition(train, y_train, estimator, params, split["model_seed"])
    return fitted, fitted.predict(features[te]).to_numpy(), tuning


def nested_cv(
    features: pd.DataFrame,
    time: Any,
    event: Any,
    estimator: str = "coxnet",
    outer_splits: int = 5,
    inner_splits: int = 3,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    seed: int = 0,
    *,
    repeats: int = 1,
    genes: tuple[str, ...] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    splits: list[dict[str, Any]] | None = None,
    permutation_repeats: int = 0,
    progress: Callable[[FoldResult], None] | None = None,
) -> CVResult:
    """Run repeated stratified nested CV with named, frozen training-only transforms.

    ``splits`` permits the identical precommitted plan across estimators. The
    historical API remains supported; result exports now include outer risks,
    inner search records, coefficients and training statistics. No model is saved.
    """
    if estimator not in ESTIMATORS:
        raise ModelingError(f"unknown estimator {estimator!r}")
    if estimator == "smith20_cox":
        genes = tuple(load_bundled("smith20").genes)
    features, structured = molecular_data(features, time, event, genes)
    plans = (
        splits
        if splits is not None
        else plan_cv(
            list(features.columns), structured["event"], outer_splits, inner_splits, repeats, seed
        )
    )
    if candidates is None:
        candidates = (
            [{"alpha": a, "l1_ratio": 0.9} for a in alphas] if estimator == "coxnet" else [{}]
        )
    result = CVResult(
        estimator,
        params={
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "repeats": repeats,
            "seed": seed,
            "n_genes": features.shape[0],
            "n_samples": features.shape[1],
            "n_events": int(structured["event"].sum()),
            "candidates": candidates,
            "split_hash": hash_json(plans),
            "genes": list(features.index),
            "permutation_repeats": permutation_repeats,
        },
        splits=plans,
    )
    pos = {s: i for i, s in enumerate(features.columns)}
    for split in plans:
        fitted, risk, tuning = evaluate_fold(features, structured, estimator, split, candidates)
        test = split["test"]
        y_test = structured[[pos[s] for s in test]]
        c = _c_index(y_test, risk)
        fold = FoldResult(
            split["fold"],
            estimator,
            fitted.alpha,
            fitted.n_train,
            len(test),
            int(y_test["event"].sum()),
            c,
            fitted.n_selected,
            fitted.scaler.checksum,
            split["repeat"],
            fitted.selected_params,
        )
        result.folds.append(fold)
        key = {"repeat": split["repeat"], "fold": split["fold"]}
        result.tuning.extend({**key, **row} for row in tuning)
        result.training.append(
            {
                **key,
                **fitted.to_dict(),
                "scaler": fitted.scaler.to_dict(),
                "warnings": list(fitted.warnings),
            }
        )
        result.predictions.extend(
            {
                **key,
                "patient": patient,
                "time": float(y["time"]),
                "event": int(y["event"]),
                "risk": float(value),
            }
            for patient, y, value in zip(test, y_test, risk, strict=True)
        )
        coef = getattr(fitted.model, "coef_", None)
        if coef is not None:
            names = ["fixed_smith20_score"] if estimator == "smith20_cox" else features.index
            result.coefficients.extend(
                {**key, "feature": str(g), "coefficient": float(b)}
                for g, b in zip(names, np.asarray(coef).ravel(), strict=True)
            )
        if estimator == "rsf" and permutation_repeats:
            rng = np.random.default_rng(split["model_seed"] + 30000)
            for gene in features.index:
                for permutation in range(permutation_repeats):
                    changed = features[test].copy()
                    changed.loc[gene] = rng.permutation(changed.loc[gene].to_numpy())
                    score = _c_index(y_test, fitted.predict(changed).to_numpy())
                    result.importance.append(
                        {
                            **key,
                            "gene": str(gene),
                            "permutation": permutation + 1,
                            "c_drop": c - score,
                        }
                    )
        if progress is not None:
            progress(fold)
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
    composite_mean: float = 0.0
    composite_sd: float = 1.0
    selected_params: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

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
        values = scaled.T.to_numpy(dtype=float)
        if not np.isfinite(matrix.loc[list(self.scaler.genes)].to_numpy(dtype=float)).all():
            raise ModelingError("required expression values must be finite")
        if self.estimator == "smith20_cox":
            values = ((values.mean(axis=1) - self.composite_mean) / self.composite_sd)[:, None]
        risk = _risk_scores(self.model, values)
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
            "selected_params": self.selected_params,
            "composite_mean": self.composite_mean,
            "composite_sd": self.composite_sd,
            "backend_detail": getattr(self.model, "detail", None),
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
    if estimator == "smith20_cox":
        ordered, outcomes = molecular_data(
            features, time, event, tuple(load_bundled("smith20").genes)
        )
        return fit_partition(ordered, outcomes, estimator, {}, seed)
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
