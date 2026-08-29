"""Compute-cost model for a compiled RailNet linear (spec 11 / open problem #2).

Dense linear ``Y = W @ x`` for one token, ``W`` shape ``(out, in)``:
    out*in multiplies + out*in adds

RailNet ``Y[j] = Σ_r R_r · (Σ_i sign(i,j,r)·X[i])``:
    * sign-weighted accumulation into G[j, r]:  ~ out*in*avg_terms  adds  (sign is ±1, no multiply)
    * rail multiply-reduce over G:              out*rail_count      multiplies + out*rail_count adds

So RailNet trades multiplies for adds. It does **not** reduce total arithmetic
and does **not** reduce weight-sized memory traffic (the route-id map is one
uint16 per weight). "Fewer multiplies" is not "faster" — that depends on the
mul:add cost ratio and whether the kernel is compute- or memory-bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _avg_terms(artifact_path: Path, route_map_path: Path) -> tuple[float, int, int, int]:
    art = json.loads(artifact_path.read_text())
    route_terms = {int(k): len(v) for k, v in art["routes"].items()}
    ids = np.load(route_map_path)
    terms = np.array([route_terms[int(b)] for b in np.unique(ids)])
    _, inv = np.unique(ids, return_inverse=True)
    per_elt_mean = float(terms[inv].mean())
    out_f, in_f = art["shape"]
    return per_elt_mean, int(art["rail_count"]), int(out_f), int(in_f)


def linear_compute(
    out_f: int, in_f: int, rail_count: int, avg_terms: float, tokens: int = 1
) -> dict:
    n = tokens
    dense_mul = n * out_f * in_f
    dense_add = n * out_f * in_f

    rail_acc_add = n * math.ceil(out_f * in_f * avg_terms)  # sign-weighted, no multiply
    rail_reduce_mul = n * out_f * rail_count
    rail_reduce_add = n * out_f * rail_count
    rail_mul = rail_reduce_mul
    rail_add = rail_acc_add + rail_reduce_add

    return {
        "avg_terms": avg_terms,
        "dense_mul": dense_mul,
        "dense_add": dense_add,
        "dense_ops": dense_mul + dense_add,
        "rail_mul": rail_mul,
        "rail_add": rail_add,
        "rail_ops": rail_mul + rail_add,
        "mul_reduction": 1 - rail_mul / dense_mul if dense_mul else 0.0,
        "add_ratio": rail_add / dense_add if dense_add else 0.0,
        "total_op_ratio": (rail_mul + rail_add) / (dense_mul + dense_add) if dense_mul else 0.0,
        # weight-sized memory: dense reads out*in BF16 weights; rail reads out*in uint16 ids.
        "weight_bytes_ratio": 1.0,
    }


def compute_cost(compiled_dir: str, tokens: int = 1) -> dict:
    out = Path(compiled_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    entries = [e for e in manifest["tensors"].values() if e.get("status") == "PASS"]

    agg = {"dense_mul": 0, "dense_add": 0, "rail_mul": 0, "rail_add": 0}
    per_tensor = {}
    for e in entries:
        avg_terms, rc, out_f, in_f = _avg_terms(out / e["artifact"], out / e["route_map"])
        c = linear_compute(out_f, in_f, rc, avg_terms, tokens)
        per_tensor[f"{e['role']}@L{e['layer']}"] = {
            "avg_terms": round(avg_terms, 3),
            "mul_reduction": round(c["mul_reduction"], 4),
            "total_op_ratio": round(c["total_op_ratio"], 4),
        }
        for k in agg:
            agg[k] += c[k]

    dm, da = agg["dense_mul"], agg["dense_add"]
    rm, ra = agg["rail_mul"], agg["rail_add"]
    return {
        "compiled_dir": str(out),
        "tensors": len(entries),
        "tokens": tokens,
        "totals": {
            **agg,
            "dense_ops": dm + da,
            "rail_ops": rm + ra,
            "mul_reduction": 1 - rm / dm if dm else 0.0,
            "add_ratio": ra / da if da else 0.0,
            "total_op_ratio": (rm + ra) / (dm + da) if dm else 0.0,
        },
        "note": (
            "mul_reduction is real; total_op_ratio ~1+ and add_ratio ~2-3x. "
            "Weight-sized memory traffic is unchanged. Not a speedup claim (spec 11)."
        ),
        "per_tensor": per_tensor,
    }
