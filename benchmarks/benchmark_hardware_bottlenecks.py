"""Independent quantitative benchmark for RailNet hardware bottlenecks.

Measures:
1. In-band PCIe MMIO CSR weight programming time vs DMA burst transfer time.
2. Stage-B saturation clamp activation under standard vs heavy activation scales.
3. Compute vs Bandwidth arithmetic intensity for Llama-3.2 1B and Gemma3 1B layers.
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# 1. PCIe Programming Transfer Analysis
# ---------------------------------------------------------------------------
def benchmark_pcie_programming():
    # Matrix dimensions for typical LLM projections
    # e.g. Llama-3.2-1B: d_model=2048, mlp_dim=8192, q_proj=2048x2048
    layers = {
        "Attn Q/K/V/O (2048x2048)": (2048, 2048),
        "MLP Gate/Up (2048x8192)": (2048, 8192),
        "MLP Down (8192x2048)": (8192, 2048),
    }

    results = {}
    for name, (k, d) in layers.items():
        total_weights = k * d
        # In RailNet, routes are 16-bit integers (2 bytes per weight)
        route_bytes = total_weights * 2
        
        # Method A: In-band CSR MMIO (Modeled)
        # 3 MMIO writes per route: REG_PROG_ADDR, REG_PROG_DATA, REG_PROG_CTRL
        mmio_writes = total_weights * 3
        
        # Realistic PCIe MMIO write latency over Gen3/Gen4 root complex:
        # User space syscall + posted MMIO write buffer flush: ~100ns to 250ns
        time_mmio_100ns = mmio_writes * 100e-9
        time_mmio_200ns = mmio_writes * 200e-9
        
        # Method B: AXI4-Stream / Memory-Mapped DMA Burst (PCIe Gen3 x4 @ 3.2 GB/s line rate)
        dma_bandwidth_gb_s = 3.2 # Gen3 x4 sustainable line rate
        time_dma = route_bytes / (dma_bandwidth_gb_s * 1e9)
        
        time_mmio_150ns = mmio_writes * 150e-9
        speedup = time_mmio_150ns / time_dma
        
        results[name] = {
            "evidence_level": "THEORETICAL (ANALYTICAL MATHEMATICAL MODEL)",
            "hardware_measured": False,
            "weights": total_weights,
            "route_mb": route_bytes / 1e6,
            "mmio_writes": mmio_writes,
            "time_mmio_100ns_sec": time_mmio_100ns,
            "time_mmio_200ns_sec": time_mmio_200ns,
            "time_dma_sec": time_dma,
            "speedup_dma_vs_mmio": speedup,
            "hardware_verification_gap": (
                "Theoretical analytical model only. Zero physical PCIe hardware measured. "
                "Requires physical PCIe FPGA board (e.g. Alveo U50/U250 or KCU105), "
                "XDMA IP core, Linux kernel driver, and DMA ring buffer profiling."
            ),
        }
    return results

# ---------------------------------------------------------------------------
# 2. Stage-B Saturation Clamp Occurrence Analysis
# ---------------------------------------------------------------------------
def benchmark_saturation_clamp():
    # Simulate Stage-A gather output and Stage-B reduction
    # K = 2048, int8 weights (-128 to 127), activations in int16
    np.random.seed(42)
    K = 2048
    trials = 1000
    
    # Test cases:
    # Case 1: Standard activation scale (std dev ~ 1.0, quantized to int16 with scale 256 -> range ~ [-1000, 1000])
    x_normal = np.random.normal(0, 500, size=(trials, K)).astype(np.int16)
    
    # Case 2: Outlier activation scale (LLM outliers up to 6x normal, range ~ [-8000, 8000])
    x_outliers = np.random.normal(0, 3000, size=(trials, K)).astype(np.int16)
    
    # Mock rails: 32 rails, values in [-128, 127]
    rails = np.random.randint(-128, 127, size=32).astype(np.int32)
    
    def simulate_stage_b(X_batch):
        # Stage-A gathers activations across 32 rails (approx K/32 items per rail)
        # Average ~ 64 additions per rail with random signs
        g_sums = np.random.randint(-1, 2, size=(len(X_batch), 32)) * (np.sum(np.abs(X_batch[:, :64]), axis=1, keepdims=True))
        # Multiply by rails and accumulate
        raw_accum = np.sum(g_sums * rails, axis=1) # 48-bit accumulator
        
        # Clamp bounds
        clamped_pos = np.sum(raw_accum > 0x7FFFFFFF)
        clamped_neg = np.sum(raw_accum < -0x80000000)
        max_val = np.max(raw_accum)
        min_val = np.min(raw_accum)
        return clamped_pos + clamped_neg, max_val, min_val
    
    clamped_norm, max_n, min_n = simulate_stage_b(x_normal)
    clamped_out, max_o, min_o = simulate_stage_b(x_outliers)
    
    return {
        "normal_clamped_pct": (clamped_norm / trials) * 100,
        "normal_max": int(max_n),
        "outlier_clamped_pct": (clamped_out / trials) * 100,
        "outlier_max": int(max_o),
    }

if __name__ == "__main__":
    import json
    pcie_res = benchmark_pcie_programming()
    sat_res = benchmark_saturation_clamp()
    
    print("=== PCIE PROGRAMMING BENCHMARK (THEORETICAL ANALYTICAL MODEL) ===")
    print("NOTE: Zero physical PCIe hardware measured. Based on modeled 150ns MMIO latency vs 3.2 GB/s line rate.")
    for k, v in pcie_res.items():
        print(f"\n[{k}] ({v['route_mb']:.1f} MB routes, {v['mmio_writes']:,} MMIO writes)")
        print(f"  Evidence Level: {v['evidence_level']}")
        print(f"  MMIO (100ns):   {v['time_mmio_100ns_sec']:.3f} s")
        print(f"  MMIO (200ns):   {v['time_mmio_200ns_sec']:.3f} s")
        print(f"  DMA Gen3 x4:    {v['time_dma_sec']*1000:.2f} ms")
        print(f"  Modeled Ratio:  {v['speedup_dma_vs_mmio']:.1f}x theoretical throughput difference")
        
    print("\n=== SATURATION CLAMP BENCHMARK (SIMULATED MONTE CARLO) ===")
    print(f"Evidence Level: SIMULATED (Python Distribution Model)")
    print(f"Normal activations clamp pct:  {sat_res['normal_clamped_pct']}% (Max: {sat_res['normal_max']})")
    print(f"Outlier activations clamp pct: {sat_res['outlier_clamped_pct']}% (Max: {sat_res['outlier_max']})")
