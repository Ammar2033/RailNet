"""Route map storage — honest accounting."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_route_map(path: str, route_ids: np.ndarray):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, route_ids.astype(np.uint16))
    final = p if p.suffix == ".npy" else p.with_suffix(".npy")
    tmp.replace(final)
    return str(final)


def load_route_map(path: str) -> np.ndarray:
    return np.load(path)


def route_map_bytes(numel: int, bits_per_id: int = 16) -> int:
    return (numel * bits_per_id + 7) // 8


def honest_report(tensor_shape, rail_count: int, route_bits: int = 16):
    numel = 1
    for d in tensor_shape:
        numel *= d
    dense_bits = numel * 16  # BF16
    rbits = rail_count * 16
    route_total = numel * route_bits
    total = rbits + route_total
    return {
        "dense_bits": dense_bits,
        "rail_bits": rbits,
        "route_bits": route_total,
        "total_bits": total,
        "ratio": dense_bits / total if total else 0,
    }
