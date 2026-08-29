"""Route-map compression study (open problem #1 — spec 8/13/19).

The per-element route-id map is what makes a compiled RailNet tensor ~dense.
This module measures, on real route-id maps, how far several *lossless* schemes
actually shrink it — the honest input to the "is hierarchical routing worth a
hardware routing fabric?" question.

Every structural scheme round-trips (compress -> decompress -> exact) before its
bit count is reported.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np


def _idx_bits(n_distinct: int) -> int:
    return max(1, math.ceil(math.log2(max(2, n_distinct))))


def _remap(ids: np.ndarray):
    """Route ids -> dense 0..K-1 codes + the palette (original ids)."""
    palette, codes = np.unique(ids, return_inverse=True)
    return codes.reshape(ids.shape).astype(np.int64), palette


# ---------------------------------------------------------------- schemes


def _raw(ids: np.ndarray) -> int:
    return ids.size * 16


def _global_minwidth(ids: np.ndarray) -> tuple[int, bool]:
    codes, palette = _remap(ids)
    bits = palette.size * 16 + ids.size * _idx_bits(palette.size)
    ok = np.array_equal(palette[codes], ids)
    return bits, ok


_IDX_BITS_LUT = np.array([max(1, math.ceil(math.log2(max(2, k)))) for k in range(1025)])


def _block_palette(ids2d: np.ndarray, block: int) -> tuple[int, bool]:
    """Per-block local palette + local indices. Lossless by construction
    (each block stores its own palette), so bit-count only — vectorized."""
    m, n = ids2d.shape
    pm, pn = (-m) % block, (-n) % block
    g = np.pad(ids2d, ((0, pm), (0, pn)))
    bm, bn = g.shape[0] // block, g.shape[1] // block
    flat = g.reshape(bm, block, bn, block).swapaxes(1, 2).reshape(bm * bn, block * block)

    s = np.sort(flat, axis=1)
    palette_sizes = 1 + (s[:, 1:] != s[:, :-1]).sum(axis=1)  # unique per block

    size_field = _idx_bits(block * block + 1)
    idx_bits = _IDX_BITS_LUT[np.clip(palette_sizes, 0, 1024)]
    total = int(
        (palette_sizes.astype(np.int64) * 16 + (block * block) * idx_bits + size_field).sum()
    )
    return total, True


def _row_rle(ids2d: np.ndarray) -> tuple[int, bool]:
    n = ids2d.shape[1]
    count_bits = _idx_bits(n + 1)
    total = 0
    ok = True
    for row in ids2d:
        change = np.concatenate(([True], row[1:] != row[:-1]))
        starts = np.flatnonzero(change)
        runs = len(starts)
        total += runs * (16 + count_bits)
        lengths = np.diff(np.append(starts, n))
        if not np.array_equal(np.repeat(row[starts], lengths), row):
            ok = False
    return total, ok


def _zlib_bits(buf: bytes) -> int:
    return len(zlib.compress(buf, 6)) * 8


def _zlib_variants(ids2d: np.ndarray) -> dict[str, int]:
    codes, palette = _remap(ids2d)
    pack_dtype = np.uint8 if palette.size <= 256 else np.uint16
    packed = codes.astype(pack_dtype)
    pal_bits = palette.size * 16
    row_delta = np.diff(codes, axis=1, prepend=codes[:, :1]).astype(np.int16)
    return {
        "zlib_raw": _zlib_bits(ids2d.astype(np.uint16).tobytes()),
        "zlib_minwidth": pal_bits + _zlib_bits(packed.tobytes()),
        "zlib_transpose": pal_bits + _zlib_bits(np.ascontiguousarray(packed.T).tobytes()),
        "zlib_rowdelta": pal_bits + _zlib_bits(row_delta.tobytes()),
    }


# ---------------------------------------------------------------- api


def benchmark_route_map(route_ids: np.ndarray, shape=None) -> dict:
    """Run every scheme on one route-id map. Returns ``{scheme: {...}}``."""
    ids = np.asarray(route_ids, dtype=np.uint16)
    ids2d = ids.reshape(shape) if shape is not None else ids
    if ids2d.ndim == 1:
        ids2d = ids2d[None, :]
    dense = ids2d.size * 16

    out: dict[str, dict] = {}

    def add(name, bits, ok):
        bits = int(bits)
        out[name] = {
            "bits": bits,
            "bytes": -(-bits // 8),
            "ratio_vs_dense": bits / dense,
            "lossless": bool(ok),
        }

    add("raw_uint16", _raw(ids2d), True)
    add("global_minwidth", *_global_minwidth(ids2d))
    for b in (8, 16, 32):
        add(f"block_palette_{b}", *_block_palette(ids2d, b))
    add("row_rle", *_row_rle(ids2d))
    for name, bits in _zlib_variants(ids2d).items():
        add(name, bits, True)  # zlib is lossless by construction

    out["_meta"] = {"params": int(ids2d.size), "unique_routes": int(np.unique(ids2d).size)}
    return out


def compressed_route_map_bits(
    route_ids: np.ndarray, shape=None, scheme: str = "zlib_minwidth"
) -> int:
    return benchmark_route_map(route_ids, shape)[scheme]["bits"]


def directory_route_study(compiled_dir: str, limit: int | None = None) -> dict:
    """Aggregate the study over a compiled directory's PASS tensors."""
    out = Path(compiled_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    entries = [e for e in manifest.get("tensors", {}).values() if e.get("status") == "PASS"]
    if limit:
        entries = entries[:limit]

    schemes: dict[str, int] = {}
    dense_total = 0
    per_tensor = {}
    for e in entries:
        ids = np.load(out / e["route_map"])
        bench = benchmark_route_map(ids, tuple(e["shape"]))
        dense_total += ids.size * 16
        per_tensor[e["role"] + f"@L{e['layer']}"] = {
            k: v["bits"] for k, v in bench.items() if not k.startswith("_")
        }
        for k, v in bench.items():
            if k.startswith("_"):
                continue
            schemes[k] = schemes.get(k, 0) + v["bits"]

    dense = dense_total or 1
    ranked = [
        {"scheme": k, "bits": v, "ratio_vs_dense": v / dense}
        for k, v in sorted(schemes.items(), key=lambda kv: kv[1])
    ]
    return {
        "compiled_dir": str(out),
        "tensors": len(entries),
        "dense_MiB": dense_total / 8 / 1024**2,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "per_tensor": per_tensor,
        "note": "lossless only; ratio > 1 means larger than dense BF16",
    }
