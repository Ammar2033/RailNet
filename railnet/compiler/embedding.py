"""Embedding — exact mmap row lookup, NOT COMPRESSED per spec."""
from __future__ import annotations

import numpy as np


def compile_embedding(weight: np.ndarray):
    return {"status": "PASSTHROUGH", "note": "embedding is exact mmap row lookup, not rail-compressed", "shape": list(weight.shape)}
