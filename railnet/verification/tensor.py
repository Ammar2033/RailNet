"""Tensor-level verification."""
from __future__ import annotations

import numpy as np

from .exact import verify_tensor_exact


def verify_lossless(bits_unique: np.ndarray, table: dict, rails_bits: np.ndarray):
    r = verify_tensor_exact(bits_unique, table, rails_bits)
    assert r["lossless"], f"tensor not lossless: {r['exact']}/{r['total']}"
    return r
