"""Effective Representation Cost (spec 13).

Honest storage accounting for a compiled RailNet artifact:

    effective bits = rail table + routing table + route-id map + metadata

Never report compression from the rail count or the routing dictionary alone —
the per-element route-id map is the term that dominates and it is ~dense.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _bits_per_index(n: int) -> int:
    return max(1, math.ceil(math.log2(max(2, n))))


def tensor_cost(artifact: dict) -> dict:
    """Cost breakdown for one compiled tensor (its ``*.json`` artifact dict)."""
    shape = tuple(artifact["shape"])
    numel = math.prod(shape)
    rail_count = int(artifact["rail_count"])
    max_terms = int(artifact["max_terms"])
    routes = artifact["routes"]
    n_routes = len(routes)

    dense_bits = numel * 16  # BF16

    rail_bits = rail_count * 16

    # routing table: key (BF16 pattern, 16b) + up to max_terms * (rail index + sign)
    per_term_bits = _bits_per_index(rail_count) + 1
    route_table_bits = n_routes * (16 + max_terms * per_term_bits)

    # per-element route-id map: minimum is ceil(log2 distinct routes)/elt;
    # what the artifact actually ships is uint16.
    id_bits_min = numel * _bits_per_index(n_routes)
    id_bits_stored = numel * 16

    effective_min = rail_bits + route_table_bits + id_bits_min
    effective_stored = rail_bits + route_table_bits + id_bits_stored

    return {
        "shape": list(shape),
        "params": numel,
        "unique_routes": n_routes,
        "dense_bits": dense_bits,
        "rail_bits": rail_bits,
        "route_table_bits": route_table_bits,
        "route_id_bits_min": id_bits_min,
        "route_id_bits_stored": id_bits_stored,
        "effective_bits_min": effective_min,
        "effective_bits_stored": effective_stored,
        "ratio_min": effective_min / dense_bits,
        "ratio_stored": effective_stored / dense_bits,
    }


def representation_cost(compiled_dir: str) -> dict:
    """Aggregate Effective Representation Cost over a compiled directory.

    ``ratio_* > 1`` means RailNet's representation is *larger* than dense.
    """
    out = Path(compiled_dir)
    manifest = json.loads((out / "manifest.json").read_text())

    per_tensor = {}
    tot = {
        "dense_bits": 0,
        "rail_bits": 0,
        "route_table_bits": 0,
        "route_id_bits_min": 0,
        "route_id_bits_stored": 0,
        "effective_bits_min": 0,
        "effective_bits_stored": 0,
    }
    for name, entry in manifest.get("tensors", {}).items():
        if entry.get("status") != "PASS":
            continue
        art = json.loads((out / entry["artifact"]).read_text())
        c = tensor_cost(art)
        per_tensor[name] = c
        for k in tot:
            tot[k] += c[k]

    dense = tot["dense_bits"] or 1
    return {
        "compiled_dir": str(out),
        "tensors": len(per_tensor),
        "totals": {
            **tot,
            "ratio_min": tot["effective_bits_min"] / dense,
            "ratio_stored": tot["effective_bits_stored"] / dense,
            "dense_MiB": tot["dense_bits"] / 8 / 1024**2,
            "effective_min_MiB": tot["effective_bits_min"] / 8 / 1024**2,
            "effective_stored_MiB": tot["effective_bits_stored"] / 8 / 1024**2,
        },
        "note": (
            "ratio > 1 => larger than dense. The route-id map dominates and is ~dense; "
            "rail + routing-table bits alone are not a compression claim (spec 8/13)."
        ),
        "per_tensor": per_tensor,
    }
