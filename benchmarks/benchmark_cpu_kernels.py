"""Benchmark RailNet CPU kernels (NumPy vs Numba JIT vs C++ AVX2/AVX-512)."""

import time
import numpy as np

from railnet.kernel import CompiledTensor, rail_linear, rail_linear_fast
from railnet.runtime.linear import (
    cpp_rail_linear,
    get_available_backends,
    is_cpp_available,
    is_numba_available,
    numba_rail_linear,
)


def make_bench_tensor(
    out_features: int,
    in_features: int,
    rail_count: int = 96,
    max_terms: int = 3,
    dtype: str = "bf16",
    seed: int = 42,
) -> CompiledTensor:
    rng = np.random.RandomState(seed)
    c = CompiledTensor.__new__(CompiledTensor)
    c.checksum_ok = True
    c.tensor_name = f"bench_{out_features}x{in_features}"
    c.dtype = dtype
    c.scale = 0.0042 if dtype == "int8" else 1.0
    c.rail_count = 32 if dtype == "int8" else rail_count
    c.max_terms = max_terms
    c.shape = (out_features, in_features)
    c.out_features = out_features
    c.in_features = in_features
    c.load_seconds = 0.0

    if dtype == "int8":
        c.rails_int32 = np.array([1, 2, 3, 4, 5, 8, 16, 32] * 4)[:c.rail_count].astype(np.int32)
        c.rails_f64 = c.rails_int32.astype(np.float64)
    else:
        c.rails_f64 = rng.uniform(-0.5, 0.5, size=rail_count).astype(np.float64)
        c.rails_int32 = None

    c.term_rail = np.zeros((65536, max_terms), dtype=np.int32)
    c.term_sign = np.zeros((65536, max_terms), dtype=np.int8)
    c.term_active = np.zeros((65536, max_terms), dtype=bool)

    num_routes = min(1024, out_features * in_features)
    for g in range(num_routes):
        num_t = rng.randint(1, max_terms + 1)
        r_indices = rng.choice(c.rail_count, size=num_t, replace=False)
        signs = rng.choice([-1, 1], size=num_t)
        for t in range(num_t):
            c.term_rail[g, t] = r_indices[t]
            c.term_sign[g, t] = signs[t]
            c.term_active[g, t] = True

    c.route_ids = rng.randint(0, num_routes, size=(out_features, in_features)).astype(np.int32)
    c.prepared = False
    return c


def time_fn(fn, warmup=3, repeat=15):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return float(np.median(times))


def run_benchmarks():
    shapes = [
        ("Gemma-1B Attn (1152x1152)", 1152, 1152),
        ("Gemma-1B MLP Gate (6912x1152)", 6912, 1152),
        ("Gemma-1B MLP Down (1152x6912)", 1152, 6912),
        ("Llama-3.2-1B MLP (8192x2048)", 8192, 2048),
    ]

    print("=" * 110)
    print(" RailNet CPU Linear Kernel Benchmark (BF16 & INT8 Mixed-Precision)")
    print(f" Available Backends: {get_available_backends()}")
    print("=" * 110)

    results = []

    for name, out_f, in_f in shapes:
        print(f"\nBenchmarking {name}...")
        c_bf16 = make_bench_tensor(out_f, in_f, dtype="bf16")
        c_int8 = make_bench_tensor(out_f, in_f, dtype="int8")
        x = np.random.randn(in_f).astype(np.float32)

        # Dense reference timing (reconstructed GEMM)
        W_dummy = np.random.randn(out_f, in_f).astype(np.float32)
        dense_ms = time_fn(lambda: x @ W_dummy.T, warmup=2, repeat=10)

        # 1. NumPy Fast
        rail_linear_fast(x.astype(np.float64), c_bf16)
        numpy_ms = time_fn(lambda: rail_linear_fast(x.astype(np.float64), c_bf16), warmup=2, repeat=10)

        # 2. Numba JIT (FP32)
        numba_ms = None
        if is_numba_available():
            numba_rail_linear(x, c_bf16, precision="fp32")
            numba_ms = time_fn(lambda: numba_rail_linear(x, c_bf16, precision="fp32"), warmup=3, repeat=15)

        # 3. C++ SIMD (BF16 / FP32)
        cpp_bf16_ms = None
        if is_cpp_available():
            cpp_rail_linear(x, c_bf16, precision="fp32")
            cpp_bf16_ms = time_fn(lambda: cpp_rail_linear(x, c_bf16, precision="fp32"), warmup=3, repeat=15)

        # 4. C++ SIMD INT8 (W8A_Float)
        cpp_int8_ms = None
        if is_cpp_available():
            cpp_rail_linear(x, c_int8, precision="fp32")
            cpp_int8_ms = time_fn(lambda: cpp_rail_linear(x, c_int8, precision="fp32"), warmup=3, repeat=15)

        results.append({
            "name": name,
            "shape": f"{out_f}x{in_f}",
            "dense_ms": dense_ms,
            "numpy_ms": numpy_ms,
            "numba_ms": numba_ms,
            "cpp_bf16_ms": cpp_bf16_ms,
            "cpp_int8_ms": cpp_int8_ms,
        })

    # Print Summary Table
    print("\n" + "=" * 125)
    print(f"{'Layer / Tensor':<30} | {'Dense (ms)':<10} | {'NumPy (ms)':<10} | {'Numba (ms)':<10} | {'C++ BF16':<10} | {'C++ INT8':<10} | {'INT8 vs NP':<11} | {'INT8 vs BF16':<12}")
    print("-" * 125)

    for r in results:
        dense_str = f"{r['dense_ms']:.2f}"
        numpy_str = f"{r['numpy_ms']:.2f}"
        numba_str = f"{r['numba_ms']:.2f}" if r['numba_ms'] else "N/A"
        cpp_bf16_str = f"{r['cpp_bf16_ms']:.2f}" if r['cpp_bf16_ms'] else "N/A"
        cpp_int8_str = f"{r['cpp_int8_ms']:.2f}" if r['cpp_int8_ms'] else "N/A"
        speedup_np = f"{r['numpy_ms'] / r['cpp_int8_ms']:.1f}x" if r['cpp_int8_ms'] else "N/A"
        speedup_bf16 = f"{r['cpp_bf16_ms'] / r['cpp_int8_ms']:.2f}x" if (r['cpp_bf16_ms'] and r['cpp_int8_ms']) else "N/A"

        print(f"{r['name']:<30} | {dense_str:<10} | {numpy_str:<10} | {numba_str:<10} | {cpp_bf16_str:<10} | {cpp_int8_str:<10} | {speedup_np:<11} | {speedup_bf16:<12}")

    print("=" * 125)


if __name__ == "__main__":
    run_benchmarks()
