"""Exactness tests — verify lossless tensor reconstruction."""
import numpy as np
import pytest

from railnet.verification.exact import verify_tensor_exact, reconstruct_value
from railnet.dtypes.bf16 import float32_to_bf16_bits, bf16_bits_to_float32


class TestExactToyTensor:
    """Two rails can exactly represent values that are combinations of those rails."""

    def test_single_rail_values(self):
        rails = np.array([0x3D00, 0x3E80], dtype=np.uint16)
        table = {0x3D00: ((0, 1),), 0x3E80: ((1, 1),)}
        uniq = np.array([0x3D00, 0x3E80], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"]

    def test_negative_rail_value(self):
        # -R0 should produce the negative of rail 0
        bits_pos = float32_to_bf16_bits(0.5)
        bits_neg = float32_to_bf16_bits(-0.5)
        rails = np.array([bits_pos], dtype=np.uint16)
        table = {
            int(bits_pos): ((0, 1),),    # +R0
            int(bits_neg): ((0, -1),),   # -R0
        }
        uniq = np.array([bits_pos, bits_neg], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"]

    def test_sum_of_rails(self):
        # R0 = 0.25, R1 = 0.125 → R0+R1 = 0.375
        bits_025 = float32_to_bf16_bits(0.25)
        bits_0125 = float32_to_bf16_bits(0.125)
        bits_0375 = float32_to_bf16_bits(0.375)

        rails = np.array([bits_025, bits_0125], dtype=np.uint16)
        table = {
            int(bits_025): ((0, 1),),
            int(bits_0125): ((1, 1),),
            int(bits_0375): ((0, 1), (1, 1)),
        }
        uniq = np.array([bits_025, bits_0125, bits_0375], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"]
        assert r["exact"] == 3

    def test_difference_of_rails(self):
        # R0 = 0.5, R1 = 0.25 → R0 - R1 = 0.25
        bits_05 = float32_to_bf16_bits(0.5)
        bits_025 = float32_to_bf16_bits(0.25)

        rails = np.array([bits_05, bits_025], dtype=np.uint16)
        table = {
            int(bits_05): ((0, 1),),
            int(bits_025): ((0, 1), (1, -1)),   # R0 - R1 = 0.5 - 0.25 = 0.25 ✓
        }
        # Verify the reconstruction matches
        rails_f64 = np.array([0.5, 0.25], dtype=np.float64)
        recon = reconstruct_value(((0, 1), (1, -1)), rails_f64)
        assert float32_to_bf16_bits(recon) == int(bits_025)

    def test_zero_value(self):
        # 0.0 can be represented as +R0 - R0 if rail 0 exists
        bits_05 = float32_to_bf16_bits(0.5)
        bits_zero = float32_to_bf16_bits(0.0)

        rails = np.array([bits_05], dtype=np.uint16)
        table = {
            int(bits_05): ((0, 1),),
            int(bits_zero): ((0, 1), (0, -1)),  # +R0 - R0 = 0
        }
        uniq = np.array([bits_05, bits_zero], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"]


class TestExactReconstructionProperties:
    """Properties that the exact verification must satisfy."""

    def test_reconstruction_is_bf16_quantized(self):
        """Reconstructed value must quantize to the same BF16 bits as target."""
        # This is the fundamental guarantee
        rails_f64 = np.array([0.03125, -0.0625], dtype=np.float64)
        route = ((0, 1), (1, 1))
        recon = reconstruct_value(route, rails_f64)
        # recon = 0.03125 + (-0.0625) = -0.03125
        target_bits = float32_to_bf16_bits(-0.03125)
        recon_bits = float32_to_bf16_bits(recon)
        assert recon_bits == target_bits

    def test_verify_catches_wrong_route(self):
        """If the route doesn't reconstruct to the target bits, it's not exact."""
        bits_1 = float32_to_bf16_bits(1.0)
        bits_2 = float32_to_bf16_bits(2.0)
        rails = np.array([bits_1], dtype=np.uint16)  # Only rail: 1.0
        # Claim 2.0 is +R0, but R0=1.0 ≠ 2.0
        table = {int(bits_2): ((0, 1),)}
        uniq = np.array([bits_2], dtype=np.uint16)
        r = verify_tensor_exact(uniq, table, rails)
        assert r["lossless"] is False
        assert int(bits_2) in r["fails"]
