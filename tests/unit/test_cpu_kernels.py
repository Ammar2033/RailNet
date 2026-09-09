"""Unit tests for high-speed C++/SIMD and Numba JIT CPU inference kernels."""

import numpy as np
import pytest

from railnet.kernel import CompiledTensor, rail_linear, rail_linear_fast
from railnet.runtime.linear import (
    cpp_rail_linear,
    get_available_backends,
    is_cpp_available,
    is_numba_available,
    numba_rail_linear,
    rail_linear_dispatch,
)


def _make_synthetic_compiled(
    out_features: int = 32,
    in_features: int = 64,
    rail_count: int = 96,
    max_terms: int = 3,
    num_routes: int = 128,
    seed: int = 42,
) -> CompiledTensor:
    rng = np.random.RandomState(seed)

    c = CompiledTensor.__new__(CompiledTensor)
    c.checksum_ok = True
    c.tensor_name = "synthetic_linear"
    c.rail_count = rail_count
    c.max_terms = max_terms
    c.shape = (out_features, in_features)
    c.out_features = out_features
    c.in_features = in_features
    c.load_seconds = 0.0

    # Random BF16-like rail values
    c.rails_f64 = rng.uniform(-0.5, 0.5, size=rail_count).astype(np.float64)

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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class TestCpuKernels:
    def test_backends_discovery(self):
        backends = get_available_backends()
        assert "numpy" in backends
        if is_numba_available():
            assert "numba" in backends
        if is_cpp_available():
            assert "cpp" in backends

    @pytest.mark.skipif(not is_numba_available(), reason="Numba is not installed in the environment")
    def test_numba_fp64_parity(self):
        compiled = _make_synthetic_compiled(out_features=32, in_features=64, seed=101)
        rng = np.random.RandomState(42)
        x = rng.randn(64).astype(np.float64)

        y_ref = rail_linear(x, compiled)
        y_fast = rail_linear_fast(x, compiled)
        np.testing.assert_allclose(y_ref, y_fast, rtol=1e-12, atol=1e-12)

        y_numba = numba_rail_linear(x, compiled, precision="fp64")
        np.testing.assert_allclose(y_ref, y_numba, rtol=1e-10, atol=1e-10)
        assert _cosine_similarity(y_ref, y_numba) > 0.999999

    @pytest.mark.skipif(not is_numba_available(), reason="Numba is not installed in the environment")
    def test_numba_fp32_parity(self):
        compiled = _make_synthetic_compiled(out_features=48, in_features=96, seed=202)
        rng = np.random.RandomState(43)
        x = rng.randn(96).astype(np.float32)

        y_ref = rail_linear(x.astype(np.float64), compiled)
        y_numba_32 = numba_rail_linear(x, compiled, precision="fp32")

        cos_sim = _cosine_similarity(y_ref, y_numba_32)
        assert cos_sim > 0.99999, f"Cosine similarity too low: {cos_sim}"

    def test_auto_dispatch(self):
        compiled = _make_synthetic_compiled(out_features=16, in_features=32, seed=303)
        x = np.random.RandomState(44).randn(32).astype(np.float32)

        y_auto = rail_linear_dispatch(x, compiled, backend="auto")
        y_ref = rail_linear_fast(x, compiled)
        np.testing.assert_allclose(y_auto, y_ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not is_numba_available(), reason="Numba is not installed in the environment")
    def test_zero_input(self):
        compiled = _make_synthetic_compiled(out_features=20, in_features=40, seed=404)
        x = np.zeros(40, dtype=np.float32)

        y_numba = numba_rail_linear(x, compiled, precision="fp32")
        np.testing.assert_array_equal(y_numba, np.zeros(20, dtype=np.float32))

    @pytest.mark.skipif(not is_numba_available(), reason="Numba is not installed in the environment")
    def test_edge_shapes(self):
        # Single neuron
        c1 = _make_synthetic_compiled(out_features=1, in_features=32, seed=501)
        x1 = np.ones(32, dtype=np.float32)
        y1_ref = rail_linear(x1.astype(np.float64), c1)
        y1_numba = numba_rail_linear(x1, c1, precision="fp32")
        assert _cosine_similarity(y1_ref, y1_numba) > 0.99999

        # Single input feature
        c2 = _make_synthetic_compiled(out_features=32, in_features=1, seed=502)
        x2 = np.array([2.5], dtype=np.float32)
        y2_ref = rail_linear(x2.astype(np.float64), c2)
        y2_numba = numba_rail_linear(x2, c2, precision="fp32")
        assert _cosine_similarity(y2_ref, y2_numba) > 0.99999

    @pytest.mark.skipif(not is_cpp_available(), reason="C++ native library not compiled")
    def test_cpp_kernel_fp64(self):
        compiled = _make_synthetic_compiled(out_features=64, in_features=128, seed=601)
        x = np.random.RandomState(45).randn(128).astype(np.float64)

        y_ref = rail_linear(x, compiled)
        y_cpp = cpp_rail_linear(x, compiled, precision="fp64", num_threads=4)
        np.testing.assert_allclose(y_ref, y_cpp, rtol=1e-10, atol=1e-10)
        assert _cosine_similarity(y_ref, y_cpp) > 0.999999

    @pytest.mark.skipif(not is_cpp_available(), reason="C++ native library not compiled")
    def test_cpp_kernel_fp32(self):
        compiled = _make_synthetic_compiled(out_features=64, in_features=128, seed=602)
        x = np.random.RandomState(46).randn(128).astype(np.float32)

        y_ref = rail_linear(x.astype(np.float64), compiled)
        y_cpp = cpp_rail_linear(x, compiled, precision="fp32", num_threads=4)
        assert _cosine_similarity(y_ref, y_cpp) > 0.99999
