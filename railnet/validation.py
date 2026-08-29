"""Validation helpers: bf16-bit equality + fp64 diagnostics.

BF16 bit-exactness is the practical exactness tier;
fp64 deltas are diagnostics only (spec 26/27).
"""

import numpy as np

from .dtypes import bf16 as B


def bf16_bits(a):

    return B.fp32_array_to_bf16_bits(np.asarray(a, dtype=np.float32))


def diff_stats(a, b):
    """Returns (max_abs_fp64_diff_finite, bf16_mismatch_count)."""

    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))

    finite = d[np.isfinite(d)]

    maxd = float(finite.max()) if finite.size else 0.0

    mism = int(np.count_nonzero(bf16_bits(a) != bf16_bits(b)))

    return maxd, mism


def first_divergence_detail(ref, rn, extra=None):
    """
    Spec-47 style detail for the FIRST mismatching element.
    """

    br = bf16_bits(ref)

    bn = bf16_bits(rn)

    bad = np.flatnonzero(br != bn)

    if bad.size == 0:
        return None

    idx = int(bad[0])

    detail = {
        "flat_index": idx,
        "ref_bf16": f"0x{int(br[idx]):04X}",
        "railnet_bf16": f"0x{int(bn[idx]):04X}",
        "ref_f64": float(ref.reshape(-1)[idx]),
        "railnet_f64": float(rn.reshape(-1)[idx]),
        "fp64_delta": float(abs(ref.reshape(-1)[idx] - rn.reshape(-1)[idx])),
    }

    if extra:
        detail.update(extra)

    return detail
