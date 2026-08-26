"""
Learner — delegates to proven learned-basis compiler (04_...).
Provides dtype-generic entrypoint; currently BF16 PROVEN.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent.parent


def _load_proven():
    spec = importlib.util.spec_from_file_location("rn_learner_proven", str(_HERE / "04_bf16_learned_basis.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_PROVEN = _load_proven()


def learn_basis(values_f64: np.ndarray, bits: np.ndarray, counts: np.ndarray, rail_count: int, max_terms: int = 4, dtype: str = "bf16") -> dict:
    if dtype != "bf16":
        raise NotImplementedError(f"learn_basis for dtype={dtype} is PLANNED — only bf16 PROVEN")
    return _PROVEN.learn_basis(values_f64, bits, counts, rail_count, max_terms)


def initialize_rails(values_f64, bits, counts, rail_count):
    return _PROVEN.initialize_rails(values_f64, bits, counts, rail_count)


def analyze_unique(raw: np.ndarray):
    return _PROVEN.analyze_unique_values(raw)
