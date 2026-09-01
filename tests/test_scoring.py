"""Tests for scoring strategies and their invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypoxiapipe.errors import InsufficientGenesError, ScoringError
from hypoxiapipe.scoring import score
from hypoxiapipe.signatures import registry


@pytest.fixture
def matrix() -> pd.DataFrame:
    """Deterministic genes x samples matrix covering smith20 plus noise genes."""
    rng = np.random.default_rng(42)
    genes = list(registry.load_bundled("smith20").genes) + [f"NOISE{i}" for i in range(30)]
    return pd.DataFrame(
        rng.normal(8.0, 2.0, size=(len(genes), 60)),
        index=genes,
        columns=[f"S{i:03d}" for i in range(60)],
    )


@pytest.fixture
def sig_smith():
    return registry.load_bundled("smith20")


@pytest.fixture
def sig_yang():
    return registry.load_bundled("yang28")


class TestRowMean:
    def test_returns_one_score_per_sample(self, matrix, sig_smith) -> None:
        r = score(matrix, sig_smith, method="rowmean")
        assert len(r.scores) == matrix.shape[1]
        assert list(r.scores.index) == list(matrix.columns)

    def test_is_invariant_to_gene_row_order(self, matrix, sig_smith) -> None:
        a = score(matrix, sig_smith, method="rowmean").scores
        shuffled = matrix.sample(frac=1.0, random_state=7)
        b = score(shuffled, sig_smith, method="rowmean").scores
        pd.testing.assert_series_equal(a, b)

    def test_is_invariant_to_per_gene_linear_rescaling(self, matrix, sig_smith) -> None:
        """Per-gene z-scoring means multiplying a gene by a constant changes nothing."""
        a = score(matrix, sig_smith, method="rowmean").scores
        rescaled = matrix.copy()
        rescaled.loc["ALDOA"] = rescaled.loc["ALDOA"] * 1000 + 5
        b = score(rescaled, sig_smith, method="rowmean").scores
        pd.testing.assert_series_equal(a, b, atol=1e-10)

    def test_scores_are_approximately_centred(self, matrix, sig_smith) -> None:
        r = score(matrix, sig_smith, method="rowmean")
        assert abs(float(r.scores.mean())) < 1e-9

    def test_ignores_genes_outside_the_signature(self, matrix, sig_smith) -> None:
        a = score(matrix, sig_smith, method="rowmean").scores
        extended = pd.concat([matrix, matrix.loc[["NOISE0"]].rename(index={"NOISE0": "EXTRA"})])
        b = score(extended, sig_smith, method="rowmean").scores
        pd.testing.assert_series_equal(a, b)


class TestMedianZ:
    def test_differs_from_rowmean(self, matrix, sig_smith) -> None:
        """The legacy method is dominated by high-expression genes."""
        skewed = matrix.copy()
        skewed.loc["ALDOA"] = skewed.loc["ALDOA"] * 50
        a = score(skewed, sig_smith, method="rowmean").scores
        b = score(skewed, sig_smith, method="median_z").scores
        assert float(np.corrcoef(a, b)[0, 1]) < 0.999


class TestWeighted:
    def test_requires_coefficients(self, matrix, sig_smith) -> None:
        with pytest.raises(ScoringError, match="no coefficients"):
            score(matrix, sig_smith, method="weighted")

    def test_equals_scaled_rowmean_when_all_coefficients_equal(self) -> None:
        """Key invariant: uniform weights reduce to the unweighted mean (times n)."""
        genes = ["G1", "G2", "G3", "G4"]
        coefs = dict.fromkeys(genes, 1.0)
        spec = registry.Signature(
            name="uniform",
            genes=tuple(genes),
            scoring="weighted",
            coefficients=coefs,
            checksum=registry.compute_checksum(genes, coefs),
        )
        rng = np.random.default_rng(1)
        m = pd.DataFrame(
            rng.normal(size=(4, 20)), index=genes, columns=[f"S{i}" for i in range(20)]
        )
        w = score(m, spec, method="weighted").scores
        u = score(m, spec, method="rowmean").scores
        np.testing.assert_allclose(w.to_numpy(), u.to_numpy() * len(genes), atol=1e-10)

    def test_opposing_coefficients_cancel(self) -> None:
        """Why Yang must be scored weighted: mixed directions cancel unweighted."""
        genes = ["UP1", "UP2", "DOWN1", "DOWN2"]
        coefs = {"UP1": 1.0, "UP2": 1.0, "DOWN1": -1.0, "DOWN2": -1.0}
        spec = registry.Signature(
            name="mixed",
            genes=tuple(genes),
            scoring="weighted",
            coefficients=coefs,
            checksum=registry.compute_checksum(genes, coefs),
        )
        base = np.linspace(0, 10, 20)
        m = pd.DataFrame([base] * 4, index=genes, columns=[f"S{i}" for i in range(20)])
        w = score(m, spec, method="weighted").scores
        np.testing.assert_allclose(w.to_numpy(), np.zeros(20), atol=1e-10)

    def test_yang_defaults_to_weighted(self, sig_yang) -> None:
        assert sig_yang.scoring == "weighted"


class TestProvenanceAndCoverage:
    def test_result_carries_signature_checksum(self, matrix, sig_smith) -> None:
        r = score(matrix, sig_smith)
        assert r.checksum == sig_smith.checksum

    def test_reports_missing_genes_and_coverage(self, matrix, sig_smith) -> None:
        reduced = matrix.drop(index=["ALDOA", "ANLN"])
        r = score(reduced, sig_smith)
        assert r.n_found == 18
        assert set(r.missing) == {"ALDOA", "ANLN"}
        assert r.coverage == pytest.approx(18 / 20)

    def test_raises_when_too_few_genes_present(self, sig_smith) -> None:
        tiny = pd.DataFrame(
            np.random.default_rng(0).normal(size=(2, 10)),
            index=["ALDOA", "ANLN"],
            columns=[f"S{i}" for i in range(10)],
        )
        with pytest.raises(InsufficientGenesError):
            score(tiny, sig_smith, min_genes=3)

    def test_unknown_method_raises(self, matrix, sig_smith) -> None:
        with pytest.raises(ScoringError, match="unknown scoring method"):
            score(matrix, sig_smith, method="nonsense")


class TestEdgeCases:
    def test_zero_variance_gene_is_dropped_not_propagated(self, matrix, sig_smith) -> None:
        m = matrix.copy()
        m.loc["ALDOA"] = 5.0
        r = score(m, sig_smith, method="rowmean")
        assert r.scores.notna().all()

    def test_single_sample_matrix_raises(self, sig_smith) -> None:
        one = pd.DataFrame(np.ones((20, 1)), index=list(sig_smith.genes), columns=["only"])
        with pytest.raises(ScoringError, match="at least 2 samples"):
            score(one, sig_smith)
