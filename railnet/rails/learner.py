"""
Learner — delegates to proven learned-basis compiler (04_...).
Provides dtype-generic entrypoint; currently BF16 PROVEN.
"""
from __future__ import annotations

from ._optimize import learn_basis as _learn_basis
from ._init import initialize_rails as _initialize_rails
from ._analysis import analyze_unique_values as _analyze_unique

def learn_basis(values_f64: np.ndarray, bits: np.ndarray, counts: np.ndarray, rail_count: int, max_terms: int = 4, dtype: str = "bf16") -> dict:
    if dtype != "bf16":
        raise NotImplementedError(f"learn_basis for dtype={dtype} is PLANNED — only bf16 PROVEN")
    return _learn_basis(values_f64, bits, counts, rail_count, max_terms)

def initialize_rails(values_f64, bits, counts, rail_count):
    return _initialize_rails(values_f64, bits, counts, rail_count)

def analyze_unique(raw: np.ndarray):
    return _analyze_unique(raw)
