"""Kernel verification — ``rail_linear`` vs a dense linear, BF16-bitwise."""

from __future__ import annotations

import numpy as np

from railnet.dtypes.bf16 import fp32_array_to_bf16_bits


def verify_kernel(x: np.ndarray, dense_w: np.ndarray, compiled) -> dict:
    """``x``: (in_features,) · ``dense_w``: (out_features, in_features) float —
    the weight in the same orientation as the safetensors / route-id map."""
    from railnet.kernel import rail_linear

    y_rail = rail_linear(np.asarray(x, dtype=np.float64), compiled)
    y_dense = np.asarray(dense_w, dtype=np.float64) @ np.asarray(x, dtype=np.float64)
    ok = np.array_equal(
        fp32_array_to_bf16_bits(y_rail.astype(np.float32)),
        fp32_array_to_bf16_bits(y_dense.astype(np.float32)),
    )
    return {"ok": bool(ok), "max_abs": float(np.max(np.abs(y_rail - y_dense)))}
