"""Unit tests for INT8 CPU kernels (C++ AVX2, Numba JIT, NumPy parity)."""

import numpy as np
import pytest

from railnet.compiler.int8 import CANONICAL_INT8_BASIS
from railnet.kernel import CompiledTensor, rail_linear, rail_linear_fast
from railnet.runtime.linear import (
    cpp_rail_linear,
    is_cpp_available,
    is_numba_available,
    numba_rail_linear,
    rail_linear_dispatch,
)


def _make_synthetic_int8_compiled(
    out_features: int = 32,
    in_features: int = 64,
    rail_count: int = 32,
    max_terms: int = 3,
    num_routes: int = 128,
    scale: float = 0.0042,
    seed: int = 77,
) -> CompiledTensor:
    rng = np.random.RandomState(seed)

    c = CompiledTensor.__new__(CompiledTensor)
    c.checksum_ok = True
    c.tensor_name = "synthetic_int8_linear"
    c.dtype = "int8"
    c.scale = scale
    c.rail_count = rail_count
    c.max_terms = max_terms
    c.shape = (out_features, in_features)
    c.out_features = out_features
    c.in_features = in_features
    c.load_seconds = 0.0

    # INT8 rails
    canonical = list(CANONICAL_INT8_BASIS)[:rail_count]
    while len(canonical) < rail_count:
        canonical.append(len(canonical) + 1)
    c.rails_int32 = np.array(canonical, dtype=np.int32)
    c.rails_f64 = c.rails_int32.astype(np.float64)

    c.term_rail = np.zeros((65536, max_terms), dtype=np.int32)
    c.term_sign = np.zeros((65536, max_terms), dtype=np.int8)
    c.term_active = np.zeros((65536, max_terms), dtype=bool)

    for g in range(num_routes):
        num_t = rng.randint(1, max_terms + 1)
        r_indices = rng.choice(rail_count, size=num_t, replace=False)
        signs = rng.choice([-1, 1], size=num_t)
        for t in range(num_t):
            c.term_rail[g, t] = r_indices[t]
            c.term_sign[g, t] = signs[t]
            c.term_active[g, t] = True

    c.route_ids = rng.randint(0, num_routes, size=(out_features, in_features)).astype(np.int32)
    c.prepared = False
    return c


def test_int8_w8a_float_cpp_matches_numpy():
    c = _make_synthetic_int8_compiled(out_features=32, in_features=64, seed=101)
    rng = np.random.RandomState(202)
    x = rng.randn(64).astype(np.float32)

    y_ref = rail_linear_fast(x.astype(np.float64), c)

    if is_cpp_available():
        y_cpp = cpp_rail_linear(x, c, precision="fp32")
        np.testing.assert_allclose(y_cpp, y_ref, rtol=1e-4, atol=1e-4)


def test_int8_w8a_float_numba_matches_numpy():
    if not is_numba_available():
        pytest.skip("Numba not available")

    c = _make_synthetic_int8_compiled(out_features=32, in_features=64, seed=102)
    rng = np.random.RandomState(203)
    x = rng.randn(64).astype(np.float32)

    y_ref = rail_linear_fast(x.astype(np.float64), c)
    y_numba = numba_rail_linear(x, c, precision="fp32")

    np.testing.assert_allclose(y_numba, y_ref, rtol=1e-4, atol=1e-4)


def test_int8_w8a16_pure_integer_parity():
    c = _make_synthetic_int8_compiled(out_features=16, in_features=32, seed=103)
    rng = np.random.RandomState(204)
    x_int16 = rng.randint(-200, 200, size=32).astype(np.int16)

    # Numba integer run
    if is_numba_available():
        y_numba = numba_rail_linear(x_int16, c, precision="w8a16")
        assert y_numba.dtype == np.int32

        if is_cpp_available():
            y_cpp = cpp_rail_linear(x_int16, c, precision="w8a16")
            assert y_cpp.dtype == np.int32
            np.testing.assert_array_equal(y_cpp, y_numba)


def test_int8_linear_auto_dispatch():
    c = _make_synthetic_int8_compiled(out_features=16, in_features=32, seed=104)
    x = np.ones(32, dtype=np.float32)

    y_auto = rail_linear_dispatch(x, c, backend="auto")
    y_ref = rail_linear_fast(x.astype(np.float64), c)

    np.testing.assert_allclose(y_auto, y_ref, rtol=1e-4, atol=1e-4)
