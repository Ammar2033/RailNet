"""Exact verification primitives."""

from __future__ import annotations

import numpy as np

from railnet.dtypes.bf16 import bf16_array_to_float32, float32_to_bf16_bits


def exact_equal_bits(a: int, b: int) -> bool:
    return int(a) == int(b)


def reconstruct_value(route: tuple, rails_f64: np.ndarray) -> float:
    s = 0.0
    for rid, sign in route:
        s += int(sign) * float(rails_f64[rid])
    return s


def verify_tensor_exact(bits_unique: np.ndarray, table: dict, rails_bits: np.ndarray) -> dict:
    rails_f64 = bf16_array_to_float32(rails_bits).astype(np.float64)
    ok = 0
    fails = []
    for b in bits_unique:
        route = table.get(int(b))
        if route is None:
            fails.append(int(b))
            continue
        v = reconstruct_value(route, rails_f64)
        if int(float32_to_bf16_bits(v)) != int(b):
            fails.append(int(b))
        else:
            ok += 1
    return {
        "exact": ok,
        "total": len(bits_unique),
        "lossless": ok == len(bits_unique),
        "fails": fails,
    }
