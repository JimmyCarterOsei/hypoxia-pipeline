"""Leakage and reproducibility checks for the molecular development protocol."""

import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import concordance_index_censored

from hypoxiapipe.modeling.evaluation import paired_bootstrap
from hypoxiapipe.modeling.preprocessing import PreprocessingError
from hypoxiapipe.modeling.rbridge import FrozenCox, RBridgeError, r_available
from hypoxiapipe.modeling.train import (
    ModelingError,
    evaluate_fold,
    molecular_data,
    nested_cv,
    plan_cv,
    select_parameters,
)
from hypoxiapipe.signatures.registry import load_bundled

needs_r = pytest.mark.skipif(not r_available(), reason="R survival worker required")
KINDS = [pytest.param("smith20_cox", marks=needs_r), "coxnet", "rsf"]


@pytest.fixture
def molecular():
    rng = np.random.default_rng(271)
    genes = tuple(load_bundled("smith20").genes)
    ids = [f"P{i:03d}" for i in range(84)]
    x = pd.DataFrame(rng.normal(size=(20, 84)), index=genes, columns=ids)
    signal = x.iloc[:3].mean()
    latent = rng.exponential(size=84) * np.exp(-signal.to_numpy()) * 25
    censor = rng.uniform(12, 50, size=84)
    time = pd.Series(np.minimum(latent, censor), index=ids)
    event = pd.Series((latent < censor).astype(int), index=ids)
    return x, time, event, genes


def candidates(kind):
    if kind == "coxnet":
        return [{"alpha": 0.1, "l1_ratio": 0.5}, {"alpha": 0.02, "l1_ratio": 0.9}]
    if kind == "rsf":
        return [{"n_estimators": 8, "min_samples_leaf": leaf} for leaf in (3, 8)]
    return [{}]


@pytest.mark.parametrize("kind", KINDS)
def test_outer_perturbation_cannot_change_scaling_tuning_or_fit(molecular, kind):
    x, time, event, genes = molecular
    x, y = molecular_data(x, time, event, genes)
    split = plan_cv(list(x.columns), y["event"], 3, 2, 1, 7)[0]
    a, _, ta = evaluate_fold(x, y, kind, split, candidates(kind))
    perturbed = x.copy()
    perturbed[split["test"]] = perturbed[split["test"]] * -70 + 400
    b, _, tb = evaluate_fold(perturbed, y, kind, split, candidates(kind))
    assert a.scaler.checksum == b.scaler.checksum
    assert a.composite_mean == b.composite_mean and a.composite_sd == b.composite_sd
    assert a.selected_params == b.selected_params and ta == tb
    pd.testing.assert_series_equal(a.predict(x), b.predict(x))
    pd.testing.assert_series_equal(a.scaler.means, x[split["train"]].mean(axis=1))


@pytest.mark.parametrize("kind", KINDS)
def test_contract_enforces_order_missing_and_ignores_extra_genes(molecular, kind):
    x, time, event, genes = molecular
    x, y = molecular_data(x, time, event, genes)
    split = plan_cv(list(x.columns), y["event"], 3, 2, 1, 8)[0]
    fit, _, _ = evaluate_fold(x, y, kind, split, candidates(kind))
    expected = fit.predict(x)
    pd.testing.assert_series_equal(expected, fit.predict(x.iloc[::-1]))
    extra = pd.concat([x, pd.DataFrame([np.ones(84) * 1e9], index=["EXTRA"], columns=x.columns)])
    pd.testing.assert_series_equal(expected, fit.predict(extra))
    with pytest.raises(PreprocessingError, match="absent"):
        fit.predict(x.drop(index=genes[0]))
    with pytest.raises(ModelingError, match="missing required"):
        molecular_data(x.drop(index=genes[0]), time, event, genes)
    bad = x.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ModelingError, match="finite"):
        fit.predict(bad)


@pytest.mark.parametrize("kind", KINDS)
def test_seed_and_sample_order_reproduce_predictions_and_splits(molecular, kind):
    x, time, event, genes = molecular
    kwargs = dict(
        estimator=kind,
        outer_splits=3,
        inner_splits=2,
        repeats=2,
        seed=12,
        genes=genes,
        candidates=candidates(kind),
    )
    a = nested_cv(x, time, event, **kwargs)
    b = nested_cv(
        x.sample(frac=1, axis=1, random_state=9).iloc[::-1],
        time.sample(frac=1, random_state=2),
        event,
        **kwargs,
    )
    assert a.splits == b.splits
    assert a.predictions == b.predictions
    assert a.tuning == b.tuning
    assert len(a.predictions) == 84 * 2
    for split in a.splits:
        assert not set(split["test"]) & set(split["train"])
        for inner in split["inner"]:
            assert not set(split["test"]) & (set(inner["train"]) | set(inner["validation"]))


@needs_r
def test_baseline_is_frozen_sample_sd_gene_mean_then_composite(molecular):
    x, time, event, genes = molecular
    x, y = molecular_data(x, time, event, genes)
    split = plan_cv(list(x.columns), y["event"], 3, 2, 1, 7)[0]
    fit, risk, _ = evaluate_fold(x, y, "smith20_cox", split, [{}])
    train = x[split["train"]]
    ztrain = train.sub(train.mean(axis=1), axis=0).div(train.std(axis=1, ddof=1), axis=0)
    raw = ztrain.mean(axis=0)
    test = x[split["test"]].sub(train.mean(axis=1), axis=0).div(train.std(axis=1, ddof=1), axis=0)
    score = (test.mean(axis=0) - raw.mean()) / raw.std(ddof=1)
    np.testing.assert_allclose(risk, score * fit.model.coef_[0], atol=1e-12)
    # Using the held-out composite SD instead would produce a different risk.
    assert not np.allclose(score, (test.mean() - test.mean().mean()) / test.mean().std())


def test_labeled_outcome_mismatch_and_duplicate_samples_fail(molecular):
    x, time, event, genes = molecular
    with pytest.raises(ModelingError, match="sample IDs"):
        molecular_data(x, time.rename(index={time.index[0]: "WRONG"}), event, genes)
    with pytest.raises(ModelingError, match="duplicate"):
        molecular_data(pd.concat([x, x.iloc[:, :1]], axis=1), time, event, genes)


def test_inner_split_cannot_include_outer_test_patient(molecular):
    x, time, event, genes = molecular
    x, y = molecular_data(x, time, event, genes)
    split = plan_cv(list(x.columns), y["event"], 3, 2, 1, 7)[0]
    split["inner"][0]["train"].append(split["test"][0])
    with pytest.raises(ModelingError, match="crosses"):
        evaluate_fold(x, y, "coxnet", split, candidates("coxnet"))


@needs_r
def test_bootstrap_uses_same_patients_and_matches_harrell(molecular):
    x, time, event, genes = molecular
    a = nested_cv(x, time, event, "smith20_cox", 3, 2, seed=5, genes=genes)
    pred = pd.DataFrame(a.predictions)
    other = pred.copy()
    other["risk"] = -other.risk
    summary, paired, folds = paired_bootstrap({"a": pred, "b": other}, draws=200, seed=31)
    for (repeat, fold), group in pred.groupby(["repeat", "fold"]):
        c = concordance_index_censored(group.event.astype(bool), group.time, group.risk)[0]
        assert folds[
            (folds.model == "a") & (folds["repeat"] == repeat) & (folds.fold == fold)
        ].c_index.iloc[0] == pytest.approx(c)
    assert summary.mean_c.sum() == pytest.approx(1)
    assert paired.mean_difference.iloc[0] == pytest.approx(
        summary.mean_c.iloc[1] - summary.mean_c.iloc[0]
    )
    b = paired_bootstrap({"a": pred, "b": other}, draws=200, seed=31)
    pd.testing.assert_frame_equal(summary, b[0])
    other.loc[0, "time"] += 1
    with pytest.raises(ModelingError, match="identical"):
        paired_bootstrap({"a": pred, "b": other}, draws=200)


def test_candidate_failing_one_inner_fold_is_not_scored_on_remaining_folds(molecular, monkeypatch):
    from hypoxiapipe.modeling import train

    x, time, event, genes = molecular
    x, y = molecular_data(x, time, event, genes)
    split = plan_cv(list(x.columns), y["event"], 3, 2, 1, 7)[0]
    ids = split["train"]
    original = train.fit_partition
    calls = 0

    def fail_once(features, structured, estimator, params, seed):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelingError("deliberate convergence failure")
        return original(features, structured, estimator, params, seed)

    monkeypatch.setattr(train, "fit_partition", fail_once)
    chosen, rows = select_parameters(
        x[ids],
        y[[x.columns.get_loc(i) for i in ids]],
        "coxnet",
        candidates("coxnet"),
        split["inner"],
        7,
    )
    assert rows[0]["valid"] is False and rows[0]["mean_c"] is None
    assert chosen == candidates("coxnet")[1]


@needs_r
def test_r_frozen_cox_does_not_restandardize_the_covariate(molecular):
    x, time, event, _ = molecular
    x, y = molecular_data(x, time, event)
    z = x.mean(axis=0).to_numpy()[:, None]
    a = FrozenCox().fit(z, y)
    b = FrozenCox().fit(z * 2, y)
    assert a.coef_[0] == pytest.approx(b.coef_[0] * 2, abs=1e-9)
    np.testing.assert_allclose(a.predict(z), b.predict(z * 2), atol=1e-9)


def test_frozen_cox_has_no_python_fallback(molecular, monkeypatch):
    from hypoxiapipe.modeling import rbridge

    x, time, event, _ = molecular
    x, y = molecular_data(x, time, event)
    monkeypatch.setattr(rbridge, "r_command", lambda: None)
    with pytest.raises(RBridgeError, match="no R worker"):
        FrozenCox().fit(x.mean(axis=0).to_numpy()[:, None], y)


def test_development_refuses_an_unpinned_expression_source():
    from types import SimpleNamespace

    from hypoxiapipe.modeling.development import verify_development_cohort

    with pytest.raises(ModelingError, match="locked Phase 9B STAR"):
        verify_development_cohort(SimpleNamespace(population_hash="historical-RDS"))
