"""RailNet Numba JIT high-performance CPU inference kernel.

Zero-dependency fallback if native C++ binary is not compiled.
Uses Numba's @njit(parallel=True, fastmath=True) across output neurons.
"""

from typing import Optional
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    njit = None
    prange = range


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _numba_linear_fp32_kernel(
        x: np.ndarray,
        route_ids: np.ndarray,
        term_rail: np.ndarray,
        term_sign: np.ndarray,
        rails: np.ndarray,
        y: np.ndarray,
        out_features: int,
        in_features: int,
        rail_count: int,
        max_terms: int,
    ) -> None:
        for j in prange(out_features):
            acc = np.zeros(rail_count, dtype=np.float32)
            for i in range(in_features):
                g = route_ids[j, i]
                xi = x[i]
                for t in range(max_terms):
                    s = term_sign[g, t]
                    if s != 0:
                        r = term_rail[g, t]
                        if 0 <= r < rail_count:
                            if s > 0:
                                acc[r] += xi
                            else:
                                acc[r] -= xi
            # Stage-B: Rail accumulation
            total = np.float32(0.0)
            for r in range(rail_count):
                total += acc[r] * rails[r]
            y[j] = total

    @njit(parallel=True, fastmath=True)
    def _numba_linear_fp64_kernel(
        x: np.ndarray,
        route_ids: np.ndarray,
        term_rail: np.ndarray,
        term_sign: np.ndarray,
        rails: np.ndarray,
        y: np.ndarray,
        out_features: int,
        in_features: int,
        rail_count: int,
        max_terms: int,
    ) -> None:
        for j in prange(out_features):
            acc = np.zeros(rail_count, dtype=np.float64)
            for i in range(in_features):
                g = route_ids[j, i]
                xi = x[i]
                for t in range(max_terms):
                    s = term_sign[g, t]
                    if s != 0:
                        r = term_rail[g, t]
                        if 0 <= r < rail_count:
                            if s > 0:
                                acc[r] += xi
                            else:
                                acc[r] -= xi
            # Stage-B: Rail accumulation
            total = 0.0
            for r in range(rail_count):
                total += acc[r] * rails[r]
            y[j] = total

    @njit(parallel=True, fastmath=True)
    def _numba_linear_int8_w8a_float_kernel(
        x: np.ndarray,
        route_ids: np.ndarray,
        term_rail: np.ndarray,
        term_sign: np.ndarray,
        rails: np.ndarray,
        y: np.ndarray,
        scale: float,
        out_features: int,
        in_features: int,
        rail_count: int,
        max_terms: int,
    ) -> None:
        for j in prange(out_features):
            acc = np.zeros(rail_count, dtype=np.float32)
            for i in range(in_features):
                g = route_ids[j, i]
                xi = x[i]
                for t in range(max_terms):
                    s = term_sign[g, t]
                    if s != 0:
                        r = term_rail[g, t]
                        if 0 <= r < rail_count:
                            if s > 0:
                                acc[r] += xi
                            else:
                                acc[r] -= xi
            total = np.float32(0.0)
            for r in range(rail_count):
                total += acc[r] * np.float32(rails[r])
            y[j] = total * np.float32(scale)

    @njit(parallel=True, fastmath=True)
    def _numba_linear_int8_w8a16_kernel(
        x: np.ndarray,
        route_ids: np.ndarray,
        term_rail: np.ndarray,
        term_sign: np.ndarray,
        rails: np.ndarray,
        y: np.ndarray,
        out_features: int,
        in_features: int,
        rail_count: int,
        max_terms: int,
    ) -> None:
        for j in prange(out_features):
            acc = np.zeros(rail_count, dtype=np.int32)
            for i in range(in_features):
                g = route_ids[j, i]
                xi = np.int32(x[i])
                for t in range(max_terms):
                    s = term_sign[g, t]
                    if s != 0:
                        r = term_rail[g, t]
                        if 0 <= r < rail_count:
                            if s > 0:
                                acc[r] += xi
                            else:
                                acc[r] -= xi
            total = np.int32(0)
            for r in range(rail_count):
                total += acc[r] * np.int32(rails[r])
            y[j] = total
else:
    _numba_linear_fp32_kernel = None
    _numba_linear_fp64_kernel = None
    _numba_linear_int8_w8a_float_kernel = None
    _numba_linear_int8_w8a16_kernel = None


def is_numba_available() -> bool:
    return HAS_NUMBA


def numba_rail_linear(x: np.ndarray, compiled, precision: str = "fp32") -> np.ndarray:
    """Execute RailNet linear layer with Numba JIT multi-threading."""
    if not HAS_NUMBA:
        raise RuntimeError("Numba is not installed in the environment.")

    c = compiled
    out_f = c.out_features
    in_f = c.in_features
    routes = c.route_ids.reshape(out_f, in_f).astype(np.int32)
    is_int8 = getattr(c, "dtype", "bf16") == "int8"
    scale = float(getattr(c, "scale", 1.0))

    if precision == "w8a16":
        x_cast = np.ascontiguousarray(x, dtype=np.int16)
        rails_cast = (
            np.ascontiguousarray(c.rails_int32, dtype=np.int32)
            if getattr(c, "rails_int32", None) is not None
            else np.ascontiguousarray(c.rails_f64, dtype=np.int32)
        )
        y = np.empty(out_f, dtype=np.int32)
        _numba_linear_int8_w8a16_kernel(
            x_cast,
            routes,
            c.term_rail,
            c.term_sign,
            rails_cast,
            y,
            out_f,
            in_f,
            c.rail_count,
            c.max_terms,
        )
        return y

    if is_int8:
        x_cast = np.ascontiguousarray(x, dtype=np.float32)
        rails_cast = (
            np.ascontiguousarray(c.rails_int32, dtype=np.int32)
            if getattr(c, "rails_int32", None) is not None
            else np.ascontiguousarray(c.rails_f64, dtype=np.int32)
        )
        y = np.empty(out_f, dtype=np.float32)
        _numba_linear_int8_w8a_float_kernel(
            x_cast,
            routes,
            c.term_rail,
            c.term_sign,
            rails_cast,
            y,
            scale,
            out_f,
            in_f,
            c.rail_count,
            c.max_terms,
        )
        return y

    if precision == "fp64":
        x_cast = np.ascontiguousarray(x, dtype=np.float64)
        rails_cast = np.ascontiguousarray(c.rails_f64, dtype=np.float64)
        y = np.empty(out_f, dtype=np.float64)
        _numba_linear_fp64_kernel(
            x_cast,
            routes,
            c.term_rail,
            c.term_sign,
            rails_cast,
            y,
            out_f,
            in_f,
            c.rail_count,
            c.max_terms,
        )
        if scale != 1.0:
            y *= scale
        return y
    else:
        x_cast = np.ascontiguousarray(x, dtype=np.float32)
        rails_cast = np.ascontiguousarray(c.rails_f64, dtype=np.float32)
        y = np.empty(out_f, dtype=np.float32)
        _numba_linear_fp32_kernel(
            x_cast,
            routes,
            c.term_rail,
            c.term_sign,
            rails_cast,
            y,
            out_f,
            in_f,
            c.rail_count,
            c.max_terms,
        )
        if scale != 1.0:
            y *= np.float32(scale)
        return y
