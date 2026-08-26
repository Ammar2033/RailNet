"""Linear compiler — tensor-level entrypoint."""
from __future__ import annotations

import numpy as np

from .compiler import RailNetCompiler

_default = RailNetCompiler()


def compile_linear(raw: np.ndarray, name: str = "", dtype: str = "bf16", rails: int = 96, max_terms: int = 4):
    return _default.compile_tensor(raw, dtype=dtype, rails=rails, max_terms=max_terms)
