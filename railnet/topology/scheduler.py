"""
Scheduler — shared-computation ordering.

Shared form:
  Y[j] = Σ_r R_r * Σ_i sign(i,j,r) * X[i]

Scheduler precomputes (p_ii, p_idx, p_ss) to allow
bincount-based shared multiply.
"""

from __future__ import annotations

import numpy as np


def build_schedule(
    route_ids: np.ndarray,
    term_rail: np.ndarray,
    term_sign: np.ndarray,
    term_active: np.ndarray,
    in_features: int,
    out_features: int,
    rail_count: int,
):
    n = route_ids.size
    ii = (np.arange(n) % in_features).astype(np.int32)
    jjR = ((np.arange(n) // in_features) * rail_count).astype(np.int64)

    g = route_ids.reshape(-1)
    ii_parts, idx_parts, ss_parts = [], [], []
    max_terms = term_rail.shape[1] if term_rail.ndim == 2 else term_active.shape[1]

    for t in range(max_terms):
        act = term_active[g, t] if term_active.ndim == 2 else term_active[g * max_terms + t]
        if not np.any(act):
            continue
        sel = np.flatnonzero(act)
        ii_parts.append(ii[sel])
        idx_parts.append(jjR[sel] + term_rail[g[sel], t])
        ss_parts.append(term_sign[g[sel], t])

    if ii_parts:
        p_ii = np.concatenate(ii_parts)
        p_idx = np.concatenate(idx_parts).astype(np.int64)
        p_ss = np.concatenate(ss_parts)
    else:
        p_ii = np.zeros(0, dtype=np.int32)
        p_idx = np.zeros(0, dtype=np.int64)
        p_ss = np.zeros(0, dtype=np.int8)
    return p_ii, p_idx, p_ss
