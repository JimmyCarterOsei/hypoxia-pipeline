"""Phase 4 tests: frozen preprocessing, leakage, nested CV, and the R bridge.

The parity test is the one that matters most. Python and R computing a
concordance index from the same numbers should agree; if they drift, one of the
two has a convention this codebase is wrong about (ties, ddof, censoring), and
that is worth knowing before it reaches a table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.modeling.preprocessing import PreprocessingError, ReferenceScaler
from hypoxiapipe.modeling.rbridge import (
    RBridgeError,
    cox,
    r_available,
    results_frame,
    worker_script,
)
from hypoxiapipe.modeling.train import (
    ModelingError,
    assert_no_leakage,
    fit_final,
    nested_cv,
    to_structured,
)

needs_r = pytest.mark.skipif(not r_available(), reason="Rscript with survival+jsonlite required")


@pytest.fixture
def survival_data():
    """Return (features, time, event) with a real signal in two genes."""
    rng = np.random.default_rng(7)
    n_genes, n_samples = 30, 160
    features = pd.DataFrame(
        rng.normal(size=(n_genes, n_samples)),
        index=[f"G{i}" for i in range(n_genes)],
        columns=[f"S{i}" for i in range(n_samples)],
    )
    signal = features.loc["G0"] * 0.9 + features.loc["G1"] * 0.6
    time = np.clip(rng.exponential(scale=np.exp(-signal)) * 20, 0.5, 60)
    event = (time < 40).astype(int)
    return features, time, event


# --------------------------------------------------------------------------
# frozen preprocessing
# --------------------------------------------------------------------------


def test_transform_uses_fit_statistics_not_the_target_matrix():
    train = pd.DataFrame({"A": [1.0, 10.0], "B": [3.0, 30.0]}, index=["g1", "g2"])
    scaler = ReferenceScaler().fit(train)
    # A target whose own mean is far from the training mean must not be
    # re-centred on itself - that is the whole point of freezing.
    target = pd.DataFrame({"C": [100.0, 100.0], "D": [102.0, 102.0]}, index=["g1", "g2"])
    out = scaler.transform(target)
    expected = (100.0 - train.loc["g1"].mean()) / train.loc["g1"].std(ddof=1)
    assert out.loc["g1", "C"] == pytest.approx(expected)


def test_fitting_twice_is_an_error_not_a_silent_overwrite():
    m = pd.DataFrame(np.arange(8.0).reshape(2, 4), index=["g1", "g2"])
    scaler = ReferenceScaler().fit(m)
    with pytest.raises(PreprocessingError, match="already fitted"):
        scaler.fit(m)


def test_missing_fitted_gene_is_an_error_under_strict():
    train = pd.DataFrame(np.arange(8.0).reshape(2, 4), index=["g1", "g2"])
    scaler = ReferenceScaler().fit(train)
    partial = pd.DataFrame(np.arange(4.0).reshape(1, 4), index=["g1"])
    with pytest.raises(PreprocessingError, match="absent from the target matrix"):
        scaler.transform(partial)
    assert scaler.transform(partial, strict=False).shape[0] == 1


def test_zero_variance_gene_becomes_constant_not_infinite():
    train = pd.DataFrame([[5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]], index=["flat", "varying"])
    scaler = ReferenceScaler().fit(train)
    out = scaler.transform(train)
    assert np.isfinite(out.to_numpy()).all()
    assert (out.loc["flat"] == 0.0).all()
    assert scaler.detail["n_zero_variance"] == 1


def test_unfitted_scaler_refuses_to_transform_or_hash():
    scaler = ReferenceScaler()
    with pytest.raises(PreprocessingError, match="not fitted"):
        scaler.transform(pd.DataFrame({"a": [1.0]}, index=["g"]))
    with pytest.raises(PreprocessingError, match="not fitted"):
        _ = scaler.checksum


def test_scaler_round_trips_and_detects_tampering(tmp_path):
    train = pd.DataFrame(np.arange(12.0).reshape(3, 4), index=["a", "b", "c"])
    scaler = ReferenceScaler().fit(train)
    path = scaler.save(tmp_path / "scaler.json")
    reloaded = ReferenceScaler.load(path)
    assert reloaded.checksum == scaler.checksum
    pd.testing.assert_frame_equal(reloaded.transform(train), scaler.transform(train))

    raw = scaler.to_dict()
    raw["means"]["a"] = raw["means"]["a"] + 1.0
    with pytest.raises(PreprocessingError, match="checksum mismatch"):
        ReferenceScaler.from_dict(raw)


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


def test_training_statistics_are_independent_of_held_out_samples(survival_data):
    features, _, _ = survival_data
    train_idx = list(range(0, 120))
    test_idx = list(range(120, 160))
    assert_no_leakage(features, train_idx, test_idx)


def test_the_leakage_check_would_catch_a_scaler_fitted_on_everything(survival_data):
    features, _, _ = survival_data
    all_idx = list(range(features.shape[1]))
    test_idx = list(range(120, 160))
    # Fitting on all samples means the "training" statistics do move when the
    # held-out samples move - which is exactly what the guard must detect.
    with pytest.raises(ModelingError, match="leakage"):
        assert_no_leakage(features, all_idx, test_idx)


def test_nested_cv_folds_use_distinct_scalers(survival_data):
    features, time, event = survival_data
    result = nested_cv(features, time, event, outer_splits=4, inner_splits=2)
    checksums = {f.scaler_checksum for f in result.folds}
    assert len(checksums) == 4, "each outer fold must fit its own scaler"


# --------------------------------------------------------------------------
# nested CV
# --------------------------------------------------------------------------


def test_nested_cv_recovers_signal_and_reports_spread(survival_data):
    features, time, event = survival_data
    result = nested_cv(features, time, event, outer_splits=4, inner_splits=2)
    assert len(result.folds) == 4
    assert result.c_index_mean > 0.55  # a real signal is present
    assert result.c_index_sd > 0  # and the spread across folds is reported
    assert "C-index" in result.to_markdown()


def test_nested_cv_is_deterministic_under_a_fixed_seed(survival_data):
    features, time, event = survival_data
    a = nested_cv(features, time, event, outer_splits=4, inner_splits=2, seed=3)
    b = nested_cv(features, time, event, outer_splits=4, inner_splits=2, seed=3)
    assert a.to_dict()["fingerprint"] == b.to_dict()["fingerprint"]


def test_random_survival_forest_runs_through_the_same_protocol(survival_data):
    features, time, event = survival_data
    result = nested_cv(features, time, event, estimator="rsf", outer_splits=3, inner_splits=2)
    assert len(result.folds) == 3
    assert all(f.alpha is None for f in result.folds)


def test_too_few_events_is_refused_rather_than_fitted(survival_data):
    features, time, _ = survival_data
    almost_no_events = np.zeros(len(time), dtype=int)
    almost_no_events[:4] = 1
    with pytest.raises(ModelingError, match="too few to fit"):
        nested_cv(features, time, almost_no_events, outer_splits=5)


def test_misaligned_features_and_outcomes_are_refused(survival_data):
    features, time, event = survival_data
    with pytest.raises(ModelingError, match="genes x samples"):
        nested_cv(features.iloc[:, :100], time, event, outer_splits=3)


def test_malformed_outcomes_are_rejected():
    with pytest.raises(ModelingError, match="coded 0/1"):
        to_structured([1.0, 2.0], [0, 2])
    with pytest.raises(ModelingError, match="strictly positive"):
        to_structured([0.0, 2.0], [0, 1])


def test_final_model_carries_its_scaler_to_an_external_cohort(survival_data):
    features, time, event = survival_data
    model = fit_final(features, time, event, population_hash="sha256:abc")
    external = features.iloc[:, :20] + 3.0  # a differently-scaled cohort
    risk = model.predict(external)

    assert len(risk) == 20
    assert np.isfinite(risk).all()
    # Scoring must not depend on the external cohort's own statistics: a subset
    # scored alone gets the same risk as when scored inside the full matrix.
    full = model.predict(features + 3.0)
    np.testing.assert_allclose(risk.to_numpy(), full.iloc[:20].to_numpy(), rtol=1e-10)
    assert model.to_dict()["train_population_hash"] == "sha256:abc"


# --------------------------------------------------------------------------
# R bridge
# --------------------------------------------------------------------------


def test_worker_script_ships_with_the_package():
    assert worker_script().exists()
    assert "survival_worker" in worker_script().name


def test_unknown_action_is_rejected_before_reaching_r():
    with pytest.raises(RBridgeError, match="unknown action"):
        cox([1.0] * 20, [1] * 20, {"s": [0.0] * 20}, action="kaplan_meier")


@needs_r
def test_per_sd_cox_recovers_a_known_hazard_ratio():
    rng = np.random.default_rng(11)
    n = 400
    score = rng.normal(size=n)
    time = np.clip(rng.exponential(scale=np.exp(-0.6 * score)) * 20, 0.5, 60)
    event = (time < 40).astype(int)

    (result,) = cox(time, event, {"sig": score})
    assert result.model == "per_sd"
    assert result.hr > 1.2
    assert result.p < 0.01
    assert result.ci_low < result.hr < result.ci_high
    assert 0.5 < result.c_index < 1.0
    assert result.significant


@needs_r
def test_r_rejects_bad_input_rather_than_fitting_it():
    with pytest.raises(RBridgeError, match="refusing to fit"):
        cox([1.0] * 5, [1, 0, 1, 0, 1], {"s": [0.1] * 5})
    with pytest.raises(RBridgeError, match="event must be coded"):
        cox([1.0] * 20, [2] * 20, {"s": list(range(20))})
    with pytest.raises(RBridgeError, match="strictly positive"):
        cox([0.0] * 20, [1] * 20, {"s": list(range(20))})


@needs_r
def test_multivariable_model_adjusts_for_a_covariate():
    rng = np.random.default_rng(13)
    n = 300
    confounder = rng.normal(size=n)
    score = confounder + rng.normal(scale=0.3, size=n)  # mostly the confounder
    time = np.clip(rng.exponential(scale=np.exp(-0.8 * confounder)) * 20, 0.5, 60)
    event = (time < 40).astype(int)

    alone = cox(time, event, {"score": score})[0]
    adjusted = {
        r.name: r
        for r in cox(
            time,
            event,
            {"score": score},
            action="cox_multivariable",
            covariates={"confounder": confounder},
        )
    }
    # A score that is largely a proxy for the confounder should attenuate.
    assert adjusted["score"].hr < alone.hr
    assert set(adjusted) == {"score", "confounder"}


@needs_r
def test_quartile_analysis_reports_group_sizes():
    rng = np.random.default_rng(17)
    n = 240
    score = rng.normal(size=n)
    time = np.clip(rng.exponential(scale=np.exp(-0.7 * score)) * 20, 0.5, 60)
    event = (time < 40).astype(int)

    (result,) = cox(time, event, {"sig": score}, action="quartile")
    assert result.model == "quartile"
    assert result.detail["n_q1"] + result.detail["n_q4"] == result.n
    assert result.n < n  # the middle half is excluded by construction


@needs_r
def test_python_and_r_agree_on_the_concordance_index(survival_data):
    """R-vs-Python parity: the same numbers must give the same C-index."""
    from sksurv.metrics import concordance_index_censored  # noqa: PLC0415

    features, time, event = survival_data
    score = features.loc["G0"].to_numpy()

    (r_result,) = cox(time, event, {"sig": score})
    # R's coxph concordance is computed on the linear predictor, which for a
    # single standardised covariate with a positive coefficient is the score
    # itself; a negative coefficient flips the ordering.
    direction = 1.0 if r_result.hr > 1 else -1.0
    python_c = concordance_index_censored(
        event.astype(bool), np.asarray(time, dtype=float), direction * score
    )[0]

    assert python_c == pytest.approx(r_result.c_index, abs=0.01)


@needs_r
def test_results_frame_is_tidy(survival_data):
    features, time, event = survival_data
    results = cox(time, event, {"a": features.loc["G0"], "b": features.loc["G1"]})
    frame = results_frame(results)
    assert list(frame["name"]) == ["a", "b"]
    assert {"hr", "ci_low", "ci_high", "p", "c_index", "n_events"} <= set(frame.columns)


def test_a_length_one_array_flag_is_not_read_as_success():
    """R serialises scalars as arrays unless unboxed, and [false] is truthy."""
    from hypoxiapipe.modeling.rbridge import _scalar_true  # noqa: PLC0415

    assert _scalar_true(True) is True
    assert _scalar_true([True]) is True
    assert _scalar_true(False) is False
    assert _scalar_true([False]) is False  # the bug this guards against
    assert _scalar_true([]) is False
    assert _scalar_true(None) is False
