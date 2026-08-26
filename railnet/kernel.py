"""RailNet runtime linear kernel (bit-pattern indexed topology).

API contract (Stage-12 rule):

    output = rail_linear(x, compiled)

`compiled` carries rails + topology + this tensor's route-id
map. It NEVER contains or accepts dense weights.
"""
import importlib.util
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, str(_HERE / fname)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R12 = _load("rn_kernel_r12", "12_gemma_linear_runner.py")

CompiledTensor = R12.CompiledTensor

rail_linear = R12.rail_linear


def prepare(c):
    """
    Hoist invariant index/sign structures out of the hot loop.

    Pure restructuring: the resulting bincount accumulation
    ORDER is identical to rail_linear, so float64 results are
    bit-identical. Called once per tensor before generation.
    """
    if getattr(c, "prepared", False):

        return

    g = c.route_ids.reshape(-1)

    n = g.size

    ii = (np.arange(n) % c.in_features).astype(np.int32)

    jjR = (
        (np.arange(n) // c.in_features)
        * c.rail_count
    ).astype(np.int64)

    ii_parts = []

    idx_parts = []

    ss_parts = []

    for t in range(c.max_terms):

        act = c.term_active[g, t]

        if not np.any(act):

            continue

        sel = np.flatnonzero(act)

        ii_parts.append(ii[sel])

        idx_parts.append(
            jjR[sel]
            + c.term_rail[g[sel], t]
        )

        ss_parts.append(
            c.term_sign[g[sel], t]
        )

    if ii_parts:

        c.p_ii = np.concatenate(ii_parts)

        c.p_idx = np.concatenate(idx_parts).astype(np.int64)

        c.p_ss = np.concatenate(ss_parts)

    else:

        c.p_ii = np.zeros(0, dtype=np.int32)

        c.p_idx = np.zeros(0, dtype=np.int64)

        c.p_ss = np.zeros(0, dtype=np.int8)

    c.prepared = True


def rail_linear_fast(x, c):
    """
    Prepared fast path. Bit-identical results to rail_linear
    (same accumulation order), roughly 2x faster per call.
    """

    if not getattr(c, "prepared", False):

        prepare(c)

    xv = x[c.p_ii]

    w = c.p_ss * xv

    G = np.bincount(
        c.p_idx,
        weights=w,
        minlength=c.out_features * c.rail_count,
    )

    return (
        G.reshape(c.out_features, c.rail_count)
        * c.rails_f64[None, :]
    ).sum(axis=1)
