"""Kernel verification — rail_linear vs dense linear bitwise."""

from __future__ import annotations

import numpy as np


def verify_kernel(x: np.ndarray, dense_W: np.ndarray, compiled):
    from railnet.kernel import rail_linear

    y_rail = rail_linear(x, compiled)
    y_dense = x.astype(np.float64) @ dense_W.astype(np.float64)
    # BF16 round-trip check via bits
    from railnet.dtypes.bf16 import fp32_array_to_bf16_bits

    ok = np.array_equal(
        fp32_array_to_bf16_bits(y_rail.astype(np.float32)),
        fp32_array_to_bf16_bits(y_dense.astype(np.float32)),
    )
    return {"ok": bool(ok), "max_abs": float(np.max(np.abs(y_rail - y_dense)))}
