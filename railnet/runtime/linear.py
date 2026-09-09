"""RailNet runtime linear layer with high-speed C++/SIMD and Numba JIT auto-dispatch.

Provides unified interface:
    rail_linear_dispatch(x, compiled, backend="auto", precision="fp32", num_threads=0)

Backends:
    - "cpp": High-performance AVX2/AVX-512 OpenMP C++ SIMD kernel.
    - "numba": JIT-compiled parallel CPU gather kernel.
    - "numpy": Original bincount-based reference implementation.
    - "auto": Automatically selects CPP > Numba > NumPy.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional

import numpy as np

from railnet.kernel import CompiledTensor, prepare, rail_linear, rail_linear_fast
from railnet.runtime.numba_kernel import is_numba_available, numba_rail_linear


def _load_cpp_module():
    try:
        from railnet.csrc.builder import get_native_kernel
        return get_native_kernel()
    except Exception:
        return None


def is_cpp_available() -> bool:
    """Returns True if the native C++ SIMD library is compiled and loadable."""
    return _load_cpp_module() is not None


def get_available_backends() -> List[str]:
    """Returns a list of currently available runtime linear backends."""
    backends = []
    if is_cpp_available():
        backends.append("cpp")
    if is_numba_available():
        backends.append("numba")
    backends.append("numpy")
    return backends


def cpp_rail_linear(
    x: np.ndarray,
    compiled: CompiledTensor,
    precision: str = "fp32",
    num_threads: int = 0,
) -> np.ndarray:
    """Execute RailNet linear layer via native C++ SIMD kernel."""
    dll = _load_cpp_module()
    if dll is None:
        raise RuntimeError(
            "Native C++ RailNet kernel is not available. Please ensure a C++ compiler "
            "(MSVC cl.exe, GCC, or Clang) is installed or pre-compile railnet/csrc/rail_kernel.cpp."
        )

    c = compiled
    out_f = int(c.out_features)
    in_f = int(c.in_features)

    routes = np.ascontiguousarray(c.route_ids.reshape(-1), dtype=np.int32)
    term_r = np.ascontiguousarray(c.term_rail.reshape(-1), dtype=np.int32)
    term_s = np.ascontiguousarray(c.term_sign.reshape(-1), dtype=np.int8)

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
        status = dll.railnet_linear_int8_w8a16(
            x_cast.ctypes.data_as(ctypes.c_void_p),
            routes.ctypes.data_as(ctypes.c_void_p),
            term_r.ctypes.data_as(ctypes.c_void_p),
            term_s.ctypes.data_as(ctypes.c_void_p),
            rails_cast.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(out_f),
            ctypes.c_int64(in_f),
            ctypes.c_int32(c.rail_count),
            ctypes.c_int32(c.max_terms),
            ctypes.c_int32(num_threads),
        )
        if status != 0:
            raise RuntimeError(f"railnet_linear_int8_w8a16 returned error code {status}")
        return y

    if is_int8:
        x_cast = np.ascontiguousarray(x, dtype=np.float32)
        rails_cast = (
            np.ascontiguousarray(c.rails_int32, dtype=np.int32)
            if getattr(c, "rails_int32", None) is not None
            else np.ascontiguousarray(c.rails_f64, dtype=np.int32)
        )
        y = np.empty(out_f, dtype=np.float32)
        status = dll.railnet_linear_int8_w8a_float(
            x_cast.ctypes.data_as(ctypes.c_void_p),
            routes.ctypes.data_as(ctypes.c_void_p),
            term_r.ctypes.data_as(ctypes.c_void_p),
            term_s.ctypes.data_as(ctypes.c_void_p),
            rails_cast.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(scale),
            ctypes.c_int64(out_f),
            ctypes.c_int64(in_f),
            ctypes.c_int32(c.rail_count),
            ctypes.c_int32(c.max_terms),
            ctypes.c_int32(num_threads),
        )
        if status != 0:
            raise RuntimeError(f"railnet_linear_int8_w8a_float returned error code {status}")
        return y

    if precision == "fp64":
        x_cast = np.ascontiguousarray(x, dtype=np.float64)
        rails_cast = np.ascontiguousarray(c.rails_f64, dtype=np.float64)
        y = np.empty(out_f, dtype=np.float64)
        status = dll.railnet_linear_fp64(
            x_cast.ctypes.data_as(ctypes.c_void_p),
            routes.ctypes.data_as(ctypes.c_void_p),
            term_r.ctypes.data_as(ctypes.c_void_p),
            term_s.ctypes.data_as(ctypes.c_void_p),
            rails_cast.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(out_f),
            ctypes.c_int64(in_f),
            ctypes.c_int32(c.rail_count),
            ctypes.c_int32(c.max_terms),
            ctypes.c_int32(num_threads),
        )
        if status != 0:
            raise RuntimeError(f"railnet_linear_fp64 returned error code {status}")
        if scale != 1.0:
            y *= scale
        return y
    else:
        x_cast = np.ascontiguousarray(x, dtype=np.float32)
        rails_cast = np.ascontiguousarray(c.rails_f64, dtype=np.float32)
        y = np.empty(out_f, dtype=np.float32)
        status = dll.railnet_linear_fp32(
            x_cast.ctypes.data_as(ctypes.c_void_p),
            routes.ctypes.data_as(ctypes.c_void_p),
            term_r.ctypes.data_as(ctypes.c_void_p),
            term_s.ctypes.data_as(ctypes.c_void_p),
            rails_cast.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(out_f),
            ctypes.c_int64(in_f),
            ctypes.c_int32(c.rail_count),
            ctypes.c_int32(c.max_terms),
            ctypes.c_int32(num_threads),
        )
        if status != 0:
            raise RuntimeError(f"railnet_linear_fp32 returned error code {status}")
        if scale != 1.0:
            y *= np.float32(scale)
        return y


def rail_linear_dispatch(
    x: np.ndarray,
    compiled: CompiledTensor,
    backend: str = "auto",
    precision: str = "fp32",
    num_threads: int = 0,
) -> np.ndarray:
    """Unified linear dispatch for RailNet.

    Args:
        x: Input activation vector [in_features].
        compiled: CompiledTensor instance.
        backend: "auto", "cpp", "numba", "numpy".
        precision: "fp32" (fast SIMD) or "fp64" (bit-exact oracle).
        num_threads: Thread count for multi-threading (0 = default).
    """
    backend_choice = backend.lower()
    if backend_choice == "auto":
        env_backend = os.environ.get("RAILNET_BACKEND", "").lower()
        if env_backend:
            backend_choice = env_backend

    if backend_choice == "auto":
        if is_cpp_available():
            return cpp_rail_linear(x, compiled, precision=precision, num_threads=num_threads)
        elif is_numba_available():
            return numba_rail_linear(x, compiled, precision=precision)
        else:
            return rail_linear_fast(x, compiled)

    elif backend_choice in ("cpp", "c++", "simd"):
        return cpp_rail_linear(x, compiled, precision=precision, num_threads=num_threads)

    elif backend_choice == "numba":
        return numba_rail_linear(x, compiled, precision=precision)

    elif backend_choice in ("numpy", "reference"):
        return rail_linear_fast(x, compiled)

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {get_available_backends()}"
        )


__all__ = [
    "CompiledTensor",
    "prepare",
    "rail_linear",
    "rail_linear_fast",
    "cpp_rail_linear",
    "numba_rail_linear",
    "rail_linear_dispatch",
    "is_cpp_available",
    "is_numba_available",
    "get_available_backends",
]
