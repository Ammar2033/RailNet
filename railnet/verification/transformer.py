"""Transformer block verification — compare dense vs rail block_forward."""

from __future__ import annotations

import numpy as np

from railnet.transformer import block_forward


def verify_block(h, norms, dense_lin, rail_lin):
    h_d, _ = block_forward(h, norms, dense_lin)
    h_r, _ = block_forward(h, norms, rail_lin)
    from railnet.dtypes.bf16 import fp32_array_to_bf16_bits

    ok = np.array_equal(
        fp32_array_to_bf16_bits(h_d.astype(np.float32)),
        fp32_array_to_bf16_bits(h_r.astype(np.float32)),
    )
    return {"ok": bool(ok)}
