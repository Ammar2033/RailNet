"""Unit tests for Rail, RailBasis, and rails module."""

import numpy as np
import pytest

from railnet.rails.basis import RailBasis
from railnet.rails.rail import Rail

# ── Rail ──────────────────────────────────────────────────


class TestRail:
    def test_from_value_bf16(self):
        r = Rail.from_value(0, "bf16", 0.03125)
        assert r.id == 0
        assert r.dtype == "bf16"
        assert r.encoded_value > 0
        assert r.decoded_value == pytest.approx(0.03125, abs=1e-4)

    def test_from_value_zero(self):
        r = Rail.from_value(5, "bf16", 0.0)
        assert r.id == 5
        assert r.encoded_value == 0
        assert r.decoded_value == 0.0

    def test_from_bits(self):
        r = Rail.from_bits(1, "bf16", 0x3F80)  # 1.0
        assert r.id == 1
        assert r.encoded_value == 0x3F80
        assert r.decoded_value == pytest.approx(1.0)

    def test_from_bits_negative(self):
        r = Rail.from_bits(2, "bf16", 0xBF80)  # -1.0
        assert r.decoded_value == pytest.approx(-1.0)

    def test_frozen(self):
        r = Rail.from_value(0, "bf16", 1.0)
        with pytest.raises(AttributeError):
            r.id = 5

    def test_to_dict(self):
        r = Rail.from_value(3, "bf16", 0.5)
        d = r.to_dict()
        assert d["id"] == 3
        assert d["dtype"] == "bf16"
        assert "bits" in d
        assert "value" in d

    def test_repr(self):
        r = Rail.from_value(0, "bf16", 1.0)
        s = repr(r)
        assert "Rail(" in s
        assert "id=0" in s
        assert "bf16" in s

    def test_metadata(self):
        r = Rail.from_value(0, "bf16", 1.0, metadata={"source": "test"})
        assert r.metadata["source"] == "test"

    def test_different_dtype(self):
        r = Rail.from_value(0, "fp16", 1.0)
        assert r.dtype == "fp16"
        assert r.decoded_value == pytest.approx(1.0)


# ── RailBasis ─────────────────────────────────────────────


class TestRailBasis:
    def test_from_bits(self):
        bits = np.array([0x3F80, 0xBF80, 0x3D00], dtype=np.uint16)
        basis = RailBasis.from_bits(bits, dtype="bf16")
        assert len(basis) == 3
        assert basis.dtype_name == "bf16"

    def test_from_rail_list(self):
        rails = [
            Rail.from_value(0, "bf16", 1.0),
            Rail.from_value(1, "bf16", -1.0),
        ]
        basis = RailBasis(rails)
        assert len(basis) == 2
        assert basis.dtype_name == "bf16"

    def test_indexing(self):
        bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
        basis = RailBasis.from_bits(bits, dtype="bf16")
        assert basis[0].id == 0
        assert basis[1].id == 1

    def test_to_bits(self):
        bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
        basis = RailBasis.from_bits(bits, dtype="bf16")
        result = basis.to_bits()
        np.testing.assert_array_equal(result, bits)
        # Ensure it's a copy
        result[0] = 0
        assert basis.bits[0] != 0

    def test_values_f64(self):
        bits = np.array([0x3F80], dtype=np.uint16)  # 1.0
        basis = RailBasis.from_bits(bits, dtype="bf16")
        assert basis.values_f64[0] == pytest.approx(1.0)

    def test_to_dict(self):
        bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
        basis = RailBasis.from_bits(bits, dtype="bf16")
        d = basis.to_dict()
        assert d["dtype"] == "bf16"
        assert d["rail_count"] == 2
        assert len(d["rails"]) == 2

    def test_empty_basis_from_list(self):
        basis = RailBasis([], dtype="bf16")
        assert len(basis) == 0
