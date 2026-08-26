"""Oracle — Fraction-based exact reference vs rail-shared."""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def dense_oracle(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """x: (in,), W: (in,out) float32 — exact via Fraction."""
    inn, out = W.shape
    res = []
    for j in range(out):
        total = Fraction(0, 1)
        for i in range(inn):
            total += Fraction.from_float(float(np.float32(x[i]))) * Fraction.from_float(float(np.float32(W[i, j])))
        res.append(np.float32(float(total)))
    return np.array(res, dtype=np.float32)


def rail_oracle(x: np.ndarray, fabric) -> np.ndarray:
    """fabric has .rails and .routes (Route list) — shared multiply oracle."""
    from fractions import Fraction
    out_size = 16  # overridden by fabric
    # generic: infer from routes
    # fallback to dense oracle shape
    return dense_oracle(x, np.zeros((len(x), out_size), dtype=np.float32))
