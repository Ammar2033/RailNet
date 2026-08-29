"""Generation verification — deterministic logits exactness."""

from __future__ import annotations

import numpy as np

from railnet.dtypes.bf16 import fp32_array_to_bf16_bits


def verify_logits(dense_logits: np.ndarray, rail_logits: np.ndarray) -> dict:
    dense_bits = fp32_array_to_bf16_bits(dense_logits.astype(np.float32))
    rail_bits = fp32_array_to_bf16_bits(rail_logits.astype(np.float32))
    exact = int(np.count_nonzero(dense_bits == rail_bits))
    total = dense_bits.size
    return {
        "exact": exact,
        "total": total,
        "lossless": exact == total,
        "ratio": exact / total if total else 0,
    }
