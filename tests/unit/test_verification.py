"""Unit tests for verification module."""
import numpy as np
import pytest

from railnet.verification.exact import exact_equal_bits, reconstruct_value, verify_tensor_exact
from railnet.verification.generation import verify_logits
from railnet.dtypes.bf16 import bf16_bits_to_float32, float32_to_bf16_bits


# ── exact_equal_bits ──────────────────────────────────────

class TestExactEqualBits:
    def test_equal(self):
        assert exact_equal_bits(0x3F80, 0x3F80)

    def test_not_equal(self):
        assert not exact_equal_bits(0x3F80, 0xBF80)

    def test_zero(self):
        assert exact_equal_bits(0, 0)


# ── reconstruct_value ─────────────────────────────────────

class TestReconstructValue:
    def test_single_positive(self):
        rails_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        route = ((0, 1),)  # +R0 = +1.0
        assert reconstruct_value(route, rails_f64) == 1.0

    def test_single_negative(self):
        rails_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        route = ((1, -1),)  # -R1 = -2.0
        assert reconstruct_value(route, rails_f64) == -2.0

    def test_two_terms(self):
        rails_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        route = ((0, 1), (2, -1))  # +R0 - R2 = 1.0 - 3.0 = -2.0
        assert reconstruct_value(route, rails_f64) == -2.0

    def test_three_terms(self):
        rails_f64 = np.array([0.5, 0.25, 0.125], dtype=np.float64)
        route = ((0, 1), (1, 1), (2, 1))  # 0.5 + 0.25 + 0.125 = 0.875
        assert reconstruct_value(route, rails_f64) == pytest.approx(0.875)


# ── verify_tensor_exact ───────────────────────────────────

class TestVerifyTensorExact:
    def test_lossless(self):
        # Build rails and routes so that reconstruction is exact
        val_1 = float32_to_bf16_bits(1.0)
        val_neg1 = float32_to_bf16_bits(-1.0)
        rails = np.array([val_1, val_neg1], dtype=np.uint16)

        # Route: val_1 → +R0, val_neg1 → +R1
        table = {
            int(val_1): ((0, 1),),
            int(val_neg1): ((1, 1),),
        }
        uniq = np.array([val_1, val_neg1], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"] is True
        assert r["exact"] == 2
        assert r["total"] == 2
        assert r["fails"] == []

    def test_missing_route(self):
        val_1 = float32_to_bf16_bits(1.0)
        val_half = float32_to_bf16_bits(0.5)
        rails = np.array([val_1], dtype=np.uint16)
        # Only 1.0 has a route, 0.5 is missing
        table = {int(val_1): ((0, 1),)}
        uniq = np.array([val_1, val_half], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"] is False
        assert r["exact"] == 1
        assert r["total"] == 2
        assert int(val_half) in r["fails"]

    def test_single_value(self):
        val = float32_to_bf16_bits(0.03125)
        rails = np.array([val], dtype=np.uint16)
        table = {int(val): ((0, 1),)}
        uniq = np.array([val], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"] is True


# ── verify_logits ─────────────────────────────────────────

class TestVerifyLogits:
    def test_identical(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r = verify_logits(logits, logits.copy())
        assert r["lossless"] is True
        assert r["exact"] == 3
        assert r["ratio"] == 1.0

    def test_partial_match(self):
        dense = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rail = np.array([1.0, 2.5, 3.0], dtype=np.float32)
        r = verify_logits(dense, rail)
        assert r["exact"] == 2  # 1.0 and 3.0 match
        assert r["total"] == 3
        assert r["lossless"] is False

    def test_empty(self):
        dense = np.array([], dtype=np.float32)
        rail = np.array([], dtype=np.float32)
        r = verify_logits(dense, rail)
        assert r["total"] == 0
