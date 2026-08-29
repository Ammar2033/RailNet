"""Analytical FPGA datapath model for one linear layer (open problem #3, step 2).

Order-of-magnitude, NOT synthesis. The goal is a single question: to match a
dense MAC tile's throughput, how many DSPs / LUTs / FFs / BRAM does the RailNet
``rail_linear`` datapath need? If that trade looks favourable, Phase-1 RTL is
worth it; if not, it is not.

Coefficients are rough (Xilinx UltraScale+ ballpark) and live in ``COST`` so
they can be swapped per target.

Reference: rail_linear computes, per output token,
    G[j, r] = Σ_i sign(i,j,r)·X[i]           # stage A: ~out·in·avg_terms  ± adds
    Y[j]    = Σ_r R_r · G[j, r]               # stage B:  out·rail_count    MACs
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Per-primitive resource cost (UltraScale+ ballpark).
COST = {
    "bf16_mul_dsp": 1,  # DSPs for one BF16 multiply
    "bf16_mac_lut": 40,  # glue LUTs around a DSP MAC (unpack / round)
    "bf16_add_lut": 24,  # a BF16 adder in fabric
    "bf16_add_ff": 20,
    "route_decode_lut": 18,  # route-id -> (rail_idx, sign) per lane
    "accum_ff_per_bit": 1,
    "ctrl_lut": 400,  # fixed control / streaming per tile
}


def dense_tile(out_f: int, in_f: int, dsps: int) -> dict:
    cycles = math.ceil(out_f * in_f / dsps)
    return {
        "dsps": dsps,
        "luts": dsps * COST["bf16_mac_lut"] + COST["ctrl_lut"],
        "ffs": dsps * 48,
        "accum_bits": out_f * 32,  # FP32 accumulators, one per output
        "cycles_per_token": cycles,
        "weight_read_bits_per_token": out_f * in_f * 16,
    }


def rail_tile(
    out_f: int, in_f: int, rail_count: int, avg_terms: float, adders: int, mul_dsps: int
) -> dict:
    stage_a_ops = math.ceil(out_f * in_f * avg_terms)
    stage_a_cycles = math.ceil(stage_a_ops / adders)
    stage_b_ops = out_f * rail_count
    stage_b_cycles = math.ceil(stage_b_ops / mul_dsps)

    # G accumulators: process one output j at a time -> rail_count live accumulators.
    g_accum_bits = rail_count * 32

    luts = (
        adders * (COST["bf16_add_lut"] + COST["route_decode_lut"])
        + mul_dsps * COST["bf16_mac_lut"]
        + COST["ctrl_lut"]
    )
    return {
        "dsps": mul_dsps,
        "adders": adders,
        "luts": luts,
        "ffs": adders * COST["bf16_add_ff"] + mul_dsps * 48,
        "accum_bits": g_accum_bits + out_f * 32,
        # the two stages form a pipeline over the output stream -> max, not sum
        "cycles_per_token": max(stage_a_cycles, stage_b_cycles),
        "stage_a_cycles": stage_a_cycles,
        "stage_b_cycles": stage_b_cycles,
        # route-id map is one uint16 per weight -> same bits as the dense weight
        "route_read_bits_per_token": out_f * in_f * 16,
    }


def match_dense_throughput(
    out_f: int, in_f: int, rail_count: int, avg_terms: float, dense_dsps: int
) -> dict:
    """Size the rail datapath so its cycles/token <= the dense tile's."""
    d = dense_tile(out_f, in_f, dense_dsps)
    budget = d["cycles_per_token"]

    # pipelined stages: size each so its own cycle count fits the budget.
    adders = math.ceil(out_f * in_f * avg_terms / budget)
    mul_dsps = math.ceil(out_f * rail_count / budget)

    r = rail_tile(out_f, in_f, rail_count, avg_terms, adders, mul_dsps)
    return {
        "shape": [out_f, in_f],
        "rail_count": rail_count,
        "avg_terms": round(avg_terms, 3),
        "target_cycles_per_token": budget,
        "dense": d,
        "rail": r,
        "ratios": {
            "dsp": r["dsps"] / d["dsps"],
            "lut": r["luts"] / d["luts"],
            "ff": r["ffs"] / d["ffs"],
            "accum_bits": r["accum_bits"] / d["accum_bits"],
            "cycles": r["cycles_per_token"] / d["cycles_per_token"],
        },
    }


def model_compiled(compiled_dir: str, dense_dsps: int = 64) -> dict:
    """Apply the model to every compiled linear and aggregate the resource ratios."""
    from railnet.analysis.compute import _avg_terms

    out = Path(compiled_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    entries = [e for e in manifest["tensors"].values() if e.get("status") == "PASS"]

    tot = {"dense_dsp": 0, "rail_dsp": 0, "dense_lut": 0, "rail_lut": 0, "rail_adders": 0}
    for e in entries:
        avg_terms, rc, out_f, in_f = _avg_terms(out / e["artifact"], out / e["route_map"])
        m = match_dense_throughput(out_f, in_f, rc, avg_terms, dense_dsps)
        tot["dense_dsp"] += m["dense"]["dsps"]
        tot["rail_dsp"] += m["rail"]["dsps"]
        tot["dense_lut"] += m["dense"]["luts"]
        tot["rail_lut"] += m["rail"]["luts"]
        tot["rail_adders"] += m["rail"]["adders"]

    return {
        "compiled_dir": str(out),
        "tensors": len(entries),
        "dense_dsps_per_tile": dense_dsps,
        "totals": tot,
        "dsp_ratio": tot["rail_dsp"] / tot["dense_dsp"],
        "lut_ratio": tot["rail_lut"] / tot["dense_lut"],
        "note": (
            "Analytical, not synthesis. At matched throughput RailNet uses far fewer DSPs "
            "and more fabric (adders/LUT). Favourable only where the device is DSP-bound and "
            "the routing/gather logic (unmodelled) stays cheap."
        ),
    }
