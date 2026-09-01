"""Tests for the signature registry: hashing, verification, tamper detection."""

from __future__ import annotations

import pytest
import yaml

from hypoxiapipe.errors import ChecksumMismatchError, IncompleteSpecError, SignatureError
from hypoxiapipe.signatures import registry


class TestChecksum:
    """Canonical hashing behaviour."""

    def test_is_order_independent(self) -> None:
        a = registry.compute_checksum(["B", "A", "C"])
        b = registry.compute_checksum(["A", "B", "C"])
        assert a == b

    def test_detects_a_single_changed_gene(self) -> None:
        a = registry.compute_checksum(["A", "B", "C"])
        b = registry.compute_checksum(["A", "B", "D"])
        assert a != b

    def test_detects_a_dropped_gene(self) -> None:
        assert registry.compute_checksum(["A", "B", "C"]) != registry.compute_checksum(["A", "B"])

    def test_coefficients_change_the_hash(self) -> None:
        genes = ["A", "B"]
        assert registry.compute_checksum(genes) != registry.compute_checksum(
            genes, {"A": 0.1, "B": 0.2}
        )

    def test_coefficient_values_matter(self) -> None:
        genes = ["A", "B"]
        one = registry.compute_checksum(genes, {"A": 0.1, "B": 0.2})
        two = registry.compute_checksum(genes, {"A": 0.1, "B": 0.3})
        assert one != two

    def test_has_algorithm_prefix(self) -> None:
        assert registry.compute_checksum(["A", "B", "C"]).startswith("sha256:")


class TestBundledSpecs:
    """The shipped specs must load and verify."""

    @pytest.mark.parametrize("name", ["smith20", "ragnum32", "yang28", "toustrup15"])
    def test_loads_and_verifies(self, name: str) -> None:
        sig = registry.load_bundled(name)
        assert sig.name == name
        assert sig.n_genes > 0

    def test_expected_sizes(self) -> None:
        sizes = {
            n: registry.load_bundled(n).n_genes
            for n in ("smith20", "ragnum32", "yang28", "toustrup15")
        }
        assert sizes == {"smith20": 20, "ragnum32": 32, "yang28": 28, "toustrup15": 15}

    def test_yang_is_weighted_and_others_are_not(self) -> None:
        assert registry.load_bundled("yang28").weighted
        assert not registry.load_bundled("ragnum32").weighted

    def test_ragnum_contains_table2_marker_genes(self) -> None:
        """Guards against the category-reconstruction bug that motivated this package."""
        genes = set(registry.load_bundled("ragnum32").genes)
        assert {"ASF1B", "TDG", "UNG", "XRCC6", "HILPDA"} <= genes
        assert "MKI67" not in genes  # present in the mislabelled vector, absent from Table 2

    def test_incomplete_spec_is_rejected(self) -> None:
        with pytest.raises(IncompleteSpecError):
            registry.load_bundled("buffa51")

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(SignatureError, match="no bundled signature"):
            registry.load_bundled("does_not_exist")

    def test_list_bundled_reports_status_without_raising(self) -> None:
        items = registry.list_bundled()
        assert isinstance(items["smith20"], registry.Signature)
        assert isinstance(items["buffa51"], IncompleteSpecError)


class TestTamperDetection:
    """The core guarantee: an edited gene list fails loudly."""

    def _spec(self, tmp_path, **overrides):
        genes = ["AAA", "BBB", "CCC"]
        spec = {
            "name": "test_sig",
            "genes": genes,
            "scoring": "rowmean",
            "checksum": registry.compute_checksum(genes),
        }
        spec.update(overrides)
        p = tmp_path / "test_sig.yaml"
        p.write_text(yaml.safe_dump(spec))
        return p

    def test_valid_spec_loads(self, tmp_path) -> None:
        assert registry.load_spec(self._spec(tmp_path)).n_genes == 3

    def test_swapped_gene_is_caught(self, tmp_path) -> None:
        p = self._spec(tmp_path, genes=["AAA", "BBB", "XXX"])
        with pytest.raises(ChecksumMismatchError):
            registry.load_spec(p)

    def test_added_gene_is_caught(self, tmp_path) -> None:
        p = self._spec(tmp_path, genes=["AAA", "BBB", "CCC", "DDD"])
        with pytest.raises(ChecksumMismatchError):
            registry.load_spec(p)

    def test_missing_scoring_method_is_caught(self, tmp_path) -> None:
        """No silent default: a spec must state its published scoring method."""
        genes = ["A", "B", "C"]
        spec = {"name": "no_method", "genes": genes, "checksum": registry.compute_checksum(genes)}
        p = tmp_path / "nm.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(IncompleteSpecError, match="no 'scoring' method"):
            registry.load_spec(p)

    def test_unknown_scoring_method_is_caught(self, tmp_path) -> None:
        genes = ["A", "B", "C"]
        spec = {
            "name": "bad_method",
            "genes": genes,
            "scoring": "telepathy",
            "checksum": registry.compute_checksum(genes),
        }
        p = tmp_path / "bm.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(SignatureError, match="unknown scoring"):
            registry.load_spec(p)

    def test_weighted_without_coefficients_is_caught(self, tmp_path) -> None:
        genes = ["A", "B", "C"]
        spec = {
            "name": "w_nocoef",
            "genes": genes,
            "scoring": "weighted",
            "checksum": registry.compute_checksum(genes),
        }
        p = tmp_path / "wn.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(SignatureError, match="no coefficients"):
            registry.load_spec(p)

    def test_missing_checksum_is_caught(self, tmp_path) -> None:
        spec = {"name": "x", "genes": ["A", "B"], "scoring": "rowmean"}
        p = tmp_path / "x.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(IncompleteSpecError):
            registry.load_spec(p)

    def test_duplicate_genes_are_rejected(self, tmp_path) -> None:
        genes = ["AAA", "BBB", "AAA"]
        spec = {
            "name": "dup",
            "genes": genes,
            "scoring": "rowmean",
            "checksum": registry.compute_checksum(genes),
        }
        p = tmp_path / "dup.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(SignatureError, match="duplicate"):
            registry.load_spec(p)

    def test_coefficients_must_cover_gene_list(self, tmp_path) -> None:
        genes = ["A", "B"]
        coefs = {"A": 0.5}
        spec = {
            "name": "bad_coef",
            "genes": genes,
            "scoring": "weighted",
            "coefficients": coefs,
            "checksum": registry.compute_checksum(genes, coefs),
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(spec))
        with pytest.raises(SignatureError, match="coefficients"):
            registry.load_spec(p)
