"""Exact math oracles (level-3 verification).

``Fraction`` arithmetic with no rounding until the final BF16 cast, used to
prove that ``rail_linear`` reproduces the dense result for the *exact same*
weights — not merely "close".
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from railnet.dtypes.bf16 import bf16_bits_to_float32


def _F(x) -> Fraction:
    return Fraction(float(np.float32(x)))


def dense_oracle(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``x``: (in,) float; ``w``: (out, in) float — returns (out,) BF16-cast float32."""
    w = np.asarray(w)
    out = np.empty(w.shape[0], dtype=np.float32)
    xf = [_F(v) for v in np.asarray(x).reshape(-1)]
    for j in range(w.shape[0]):
        total = Fraction(0)
        for i, xi in enumerate(xf):
            total += xi * _F(w[j, i])
        out[j] = np.float32(float(total))
    return out


def rail_oracle(x: np.ndarray, rails_bits: np.ndarray, routes: dict, route_ids: np.ndarray):
    """Exact rail evaluation of ``Y = W @ x`` where ``W[j,i] = Σ sign·rail``.

    ``routes``: ``{bf16_bits: ((rail_id, sign), ...)}`` (0-indexed rail ids).
    ``route_ids``: (out, in) uint16 map of each weight's BF16 bit pattern.
    """
    rail_f = [_F(bf16_bits_to_float32(int(b))) for b in np.asarray(rails_bits)]
    route_ids = np.asarray(route_ids)
    out_features, in_features = route_ids.shape
    xf = [_F(v) for v in np.asarray(x).reshape(-1)]
    out = np.empty(out_features, dtype=np.float32)
    for j in range(out_features):
        total = Fraction(0)
        for i in range(in_features):
            route = routes.get(int(route_ids[j, i]))
            if route is None:
                raise KeyError(f"no route for weight ({j},{i}) bits={int(route_ids[j, i])}")
            wji = Fraction(0)
            for rid, sign in route:
                wji += int(sign) * rail_f[int(rid)]
            total += xf[i] * wji
        out[j] = np.float32(float(total))
    return out
