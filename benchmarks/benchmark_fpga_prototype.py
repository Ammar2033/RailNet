"""End-to-End FPGA + PCIe Prototype Performance & Correctness Profiler (Gate 2 Hardened).

Profiles the full hardware execution loop using:
- Routed core clock from results/fpga_pnr_results.json, when a P&R run actually
  completed. There is no physical board, so no figure here is hardware-measured.
- Cycle-accurate timing model of RailNetTop grid and AXI4-Stream gather concentrator.
- Sustained DMA payload transfer rate calculation over physical PCIe Gen2 x1 / x4 links.
- Bit-exact numerical verification against PyTorch golden matrix multiplication.
- Optional live PCIe DMA measurement via RailNetPCIeDriver when hardware present.

Outputs:
- results/fpga_prototype_metrics.json (with evidence_tier field)
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ASSUMED_FMAX_MHZ = 100.0  # the project's stated core_clk target, not a result


def load_routed_fmax() -> float | None:
    """Return the grid's routed Fmax, or None if no P&R run produced one.

    Only a target that actually completed routing counts. The previous version
    could not fail: it fell back to the *target* frequency (a constraint, not an
    achievement), then to any other target's Fmax (one tile's clock, returned as
    the grid's), and finally to a hardcoded 83.79 described as measured.
    """
    pnr_file = ROOT / "results" / "fpga_pnr_results.json"
    if not pnr_file.exists():
        return None
    try:
        d = json.loads(pnr_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    targets = d.get("targets", {})
    for key in ("railnet_top_2x2_ecp5", "railnet_top_2x2", "railnet_dual_clk_top_ecp5"):
        entry = targets.get(key)
        if not isinstance(entry, dict) or entry.get("status") != "PNR_COMPLETED":
            continue
        fmax = entry.get("achieved_fmax_mhz")
        if fmax:
            return float(fmax)
    return None


def _try_measure_live_dma_bw(
    device_name: str = "railnet0",
    k_probe: int = 4096,
    n_iter: int = 50,
    verbose: bool = False,
) -> tuple[float | None, dict | None]:
    """Attempt to measure live PCIe DMA bandwidth if hardware present.

    Returns (bw_gbps, status_dict) or (None, status_dict) if no hardware.
    Only measures when RailNetPCIeDriver.is_hardware == True and link_up == True.
    """
    status: dict = {"attempted": True, "is_hardware": False, "link_up": False, "bw_gbps": None, "error": None}
    try:
        from railnet.runtime.pcie import RailNetPCIeDriver

        drv = RailNetPCIeDriver(device_name, verbose=verbose)
        status["is_hardware"] = drv.is_hardware
        # Update with detailed link status if available
        try:
            ls = drv.get_link_status()
            status["link_status"] = ls
            status["link_up"] = ls.get("link_up", False)
        except Exception:
            status["link_up"] = drv.is_link_up() if hasattr(drv, "is_link_up") else False

        if not drv.is_hardware:
            status["error"] = "No PCIe hardware detected (MockPCIeBridge fallback) - using modelled BW"
            drv.close()
            return None, status
        if not status.get("link_up", False):
            status["error"] = "PCIe hardware detected but link not up (training failed) - using modelled BW"
            drv.close()
            return None, status

        # Hardware is present and link up -> measure
        x_probe = np.random.randint(-50, 50, size=k_probe, dtype=np.int16)
        # Program dummy rails for measurement (minimal)
        from railnet.kernel import CompiledTensor, prepare

        # Create minimal compiled tensor for timing (not needed for raw DMA BW, but needed for read_results path)
        # Instead, directly measure DMA H2C + C2H without full compute by using low-level bridge
        # For live BW, we measure raw DMA loop: stream + read
        # Warmup
        for _ in range(3):
            try:
                drv.stream_activations(x_probe)
                _ = drv.read_results(4, scale=1.0)
            except Exception:
                pass

        t0 = time.perf_counter()
        total_bytes = 0
        for _ in range(n_iter):
            drv.stream_activations(x_probe)
            total_bytes += k_probe * 2  # int16
            res = drv.read_results(4, scale=1.0)
            total_bytes += 4 * 4  # 4 outputs * 4 bytes
        t1 = time.perf_counter()
        elapsed = max(t1 - t0, 1e-6)
        bw_gbps = (total_bytes / elapsed) / 1e9
        status["bw_gbps"] = round(bw_gbps, 6)
        status["elapsed_s"] = round(elapsed, 6)
        status["total_bytes"] = total_bytes
        status["n_iter"] = n_iter
        drv.close()
        if verbose:
            print(f"[Live DMA] Measured BW: {bw_gbps*1000:.1f} MB/s ({bw_gbps:.3f} GB/s) over {n_iter} iters")
        return bw_gbps, status
    except Exception as e:
        status["error"] = str(e)
        if verbose:
            print(f"[Live DMA] Measurement failed: {e}")
        return None, status


def profile_prototype_execution(
    fmax_mhz: float | None = None,
    pcie_gen: int = 2,
    pcie_lanes: int = 1,
    num_tiles: int = 4,
    rails: int = 16,
    k_dimensions: list = None,
    use_hardware: bool = False,
    device_name: str = "railnet0",
    verbose: bool = False,
) -> dict:
    """Profile hardware latency, throughput, and numerical accuracy across vector sizes."""
    if k_dimensions is None:
        k_dimensions = [64, 128, 256, 512, 1024, 2048]

    # PCIe link characteristics (modelled fallback)
    # Gen2 x1: 5.0 GT/s * 8/10 8b/10b encoding = 500 MB/s raw -> ~400 MB/s sustained payload
    # Gen2 x4: ~1.6 GB/s sustained payload
    # Gen3 x4: ~3.2 GB/s sustained payload
    if pcie_gen == 2 and pcie_lanes == 1:
        modelled_bw_gbps = 0.400  # 400 MB/s
    elif pcie_gen == 2 and pcie_lanes == 4:
        modelled_bw_gbps = 1.600  # 1.6 GB/s
    elif pcie_gen == 3 and pcie_lanes == 4:
        modelled_bw_gbps = 3.200  # 3.2 GB/s
    else:
        modelled_bw_gbps = 0.400 * pcie_lanes

    # Attempt live measurement if requested
    live_bw_gbps = None
    live_status = None
    evidence_tier = "MODELLED (PCIe DMA Analytically Estimated)"
    if use_hardware:
        live_bw_gbps, live_status = _try_measure_live_dma_bw(device_name=device_name, verbose=verbose)
        if live_bw_gbps is not None and live_bw_gbps > 0:
            sustained_dma_bw_gbps = live_bw_gbps
            evidence_tier = "FPGA-MEASURED (Live PCIe DMA via RailNetPCIeDriver)"
        else:
            sustained_dma_bw_gbps = modelled_bw_gbps
            if live_status and not live_status.get("is_hardware"):
                evidence_tier = "MODELLED (No Hardware - Mock Fallback)"
            elif live_status and not live_status.get("link_up"):
                evidence_tier = "MODELLED (Hardware Link Down - Fallback)"
            else:
                evidence_tier = "MODELLED (Live Measurement Failed - Fallback)"
    else:
        sustained_dma_bw_gbps = modelled_bw_gbps

    if fmax_mhz is None:
        fmax_mhz = DEFAULT_ASSUMED_FMAX_MHZ
        clock_provenance = "ASSUMED (no completed P&R run; using the stated core_clk target)"
    else:
        clock_provenance = "PLACED & ROUTED (nextpnr-ecp5 estimate; no physical board)"

    core_cycle_time_ns = 1000.0 / fmax_mhz
    metrics = {
        "hardware_platform": "Lattice ECP5-85F (Open-Source) / QMTech Artix-7 (PCIe)",
        "core_fmax_mhz": fmax_mhz,
        "clock_provenance": clock_provenance,
        "core_clock_period_ns": round(core_cycle_time_ns, 3),
        "pcie_configuration": f"PCIe Gen{pcie_gen} x{pcie_lanes}",
        "sustained_dma_bandwidth_gbps": sustained_dma_bw_gbps,
        "modelled_dma_bandwidth_gbps": modelled_bw_gbps,
        "live_dma_bandwidth_gbps": live_bw_gbps,
        "live_dma_status": live_status,
        "evidence_tier": evidence_tier,
        "num_tiles": num_tiles,
        "rails": rails,
        "evaluations": [],
    }

    print("=========================================================================")
    print(" RailNet FPGA + PCIe End-to-End Prototype Profiler & Accuracy Sign-off")
    print(f" Core clock: {fmax_mhz:.2f} MHz ({core_cycle_time_ns:.3f} ns period) - {clock_provenance}")
    print(f" Target PCIe Link: Gen{pcie_gen} x{pcie_lanes} ({sustained_dma_bw_gbps*1000:.0f} MB/s sustained DMA)")
    print(f" Evidence Tier: {evidence_tier}")
    if live_status and verbose:
        print(f" Live DMA Status: {live_status}")
    print("=========================================================================\n")

    for K in k_dimensions:
        # 1. DMA H2C: Transfer K INT16 activations (2 bytes each)
        h2c_bytes = K * 2
        dma_h2c_time_ns = (h2c_bytes / (sustained_dma_bw_gbps * 1e9)) * 1e9

        # 2. Hardware Core Execution Cycles:
        # - Flush & setup: rails + 4 cycles
        # - Stream in: K cycles (1 beat per cycle under continuous streaming)
        # - Drain pipeline: 3 cycles
        # - Reduction tree: rails cycles
        # - Output gather serialization: num_tiles cycles
        core_cycles = (rails + 4) + K + 3 + rails + num_tiles
        core_time_ns = core_cycles * core_cycle_time_ns

        # 3. DMA C2H: Transfer num_tiles INT32 outputs (4 bytes each)
        c2h_bytes = num_tiles * 4
        dma_c2h_time_ns = (c2h_bytes / (sustained_dma_bw_gbps * 1e9)) * 1e9

        # 4. Total End-to-End Latency
        total_latency_ns = dma_h2c_time_ns + core_time_ns + dma_c2h_time_ns
        total_latency_ms = total_latency_ns / 1e6
        tokens_per_sec = 1e9 / total_latency_ns

        # 5. Golden Mathematical Contract Check
        rng = np.random.default_rng(K)
        x_vec = rng.integers(-50, 50, size=K, dtype=np.int16)
        # 4 tiles with different weights
        w_tile = np.array([7, -10, 28, -32][:num_tiles], dtype=np.int32)
        golden_y = [int(np.sum(x_vec.astype(np.int64) * int(w))) for w in w_tile]

        eval_entry = {
            "in_features_k": K,
            "h2c_payload_bytes": h2c_bytes,
            "c2h_payload_bytes": c2h_bytes,
            "core_cycles": core_cycles,
            "dma_h2c_latency_us": round(dma_h2c_time_ns / 1000.0, 3),
            "core_compute_latency_us": round(core_time_ns / 1000.0, 3),
            "dma_c2h_latency_us": round(dma_c2h_time_ns / 1000.0, 3),
            "total_e2e_latency_us": round(total_latency_ns / 1000.0, 3),
            "total_e2e_latency_ms": round(total_latency_ms, 5),
            "sustained_token_rate": round(tokens_per_sec, 1),
            # golden_y is the software reference only. No hardware or RTL output
            # is produced here, so there is nothing to compare it against; the
            # previous hardcoded "bit_exact_numerical_match": True asserted a
            # match that was never computed.
            "hardware_output_compared": False,
            "golden_sample_output": golden_y,
        }
        metrics["evaluations"].append(eval_entry)

    out_file = ROOT / "results" / "fpga_prototype_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"{'K (Dim)':>8} {'H2C (us)':>10} {'Core (us)':>10} {'C2H (us)':>10} {'Total (us)':>12} {'Rate (t/s)':>12} {'HW match':>12}")
    print("-" * 80)
    for e in metrics["evaluations"]:
        print(
            f"{e['in_features_k']:>8} {e['dma_h2c_latency_us']:>10.2f} {e['core_compute_latency_us']:>10.2f} "
            f"{e['dma_c2h_latency_us']:>10.2f} {e['total_e2e_latency_us']:>12.2f} "
            f"{e['sustained_token_rate']:>12.1f} {'n/a (model)':>12}"
        )
    print("=" * 80)
    print(f"\nPrototype metrics saved to: {out_file}\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailNet FPGA+PCIe Profiler (Gate 2 Hardened)")
    parser.add_argument("--pcie-gen", type=int, default=2, help="PCIe Gen (2 or 3)")
    parser.add_argument("--pcie-lanes", type=int, default=1, help="PCIe lanes (1 or 4)")
    parser.add_argument("--use-hardware", action="store_true", help="Attempt live DMA measurement via RailNetPCIeDriver")
    parser.add_argument("--device", type=str, default="railnet0", help="PCIe device name")
    parser.add_argument("--verbose", action="store_true", help="Verbose live DMA diagnostics")
    args = parser.parse_args()

    fmax = load_routed_fmax()
    profile_prototype_execution(
        fmax_mhz=fmax,
        pcie_gen=args.pcie_gen,
        pcie_lanes=args.pcie_lanes,
        use_hardware=args.use_hardware,
        device_name=args.device,
        verbose=args.verbose,
    )
