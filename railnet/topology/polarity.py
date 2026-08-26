"""Polarity helpers."""
from __future__ import annotations

import numpy as np


def polarity_matrix(signs: np.ndarray) -> np.ndarray:
    """signs: (n_unique, max_terms) int8 -> float64 polarity."""
    return signs.astype(np.float64)


def validate_polarity(signs: np.ndarray, routes: np.ndarray) -> bool:
    # signs must be 0 where route is inactive, +-1 otherwise
    active = routes > 0
    return bool(np.all(np.isin(signs[active], [-1, 1])) and np.all(signs[~active] == 0))
