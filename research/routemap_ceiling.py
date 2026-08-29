"""How compressible is the route-id map, really? (open problem #1, ceiling)

The route-id map is the BF16 weight values relabeled — `route_id[i,j]` is a pure
function of `weight_bits[i,j]`. So *any* lossless route-map codec is bounded by
the lossless compressibility of the trained weights themselves. This measures
that ceiling on the real Gemma3 1B tensors:

  * per-tensor and GLOBAL unique-route counts (cross-tensor rail sharing)
  * lossless size of the raw BF16 weights via zlib / lzma / bit-plane split
  * order-0 entropy of the weight-value distribution

If the ceiling is ~0.8x, Yol A (compress the map) cannot fund a "model in SRAM"
story and the effort should go to Yol B (dense NVM) or structured routing.
"""

from __future__ import annotations

import json
import lzma
import math
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from railnet.safetensors_reader import list_tensors, read_tensor_raw

ROOT = Path(__file__).resolve().parent.parent
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def _order0_entropy_bits(arr: np.ndarray) -> float:
    counts = np.bincount(arr.reshape(-1).astype(np.int64))
    p = counts[counts > 0] / arr.size
    return float(-(p * np.log2(p)).sum())


def _linears():
    for name in list_tensors():
        if not name.startswith("model.layers."):
            continue
        parts = name.split(".")
        sub = ".".join(parts[3:])
        role = next((r for r in ATTN if sub == f"self_attn.{r}.weight"), None) or next(
            (r for r in MLP if sub == f"mlp.{r}.weight"), None
        )
        if role:
            yield name, int(parts[2]), role


def main() -> int:
    global_vals: Counter = Counter()
    per_tensor_unique = []
    dense_bits = 0
    zlib_bits = 0
    lzma_bits = 0
    bitplane_zlib_bits = 0
    ent_weighted = 0.0

    rows = []
    for name, layer, role in _linears():
        raw, _shape = read_tensor_raw(name)  # uint16 BF16 bits
        n = raw.size
        dense_bits += n * 16
        u = np.unique(raw)
        per_tensor_unique.append(u.size)
        global_vals.update(raw.tolist())

        zlib_bits += len(zlib.compress(raw.tobytes(), 6)) * 8
        lzma_bits += len(lzma.compress(raw.tobytes(), preset=1)) * 8
        hi = (raw >> 8).astype(np.uint8).tobytes()
        lo = (raw & 0xFF).astype(np.uint8).tobytes()
        bitplane_zlib_bits += (len(zlib.compress(hi, 6)) + len(zlib.compress(lo, 6))) * 8

        ent = _order0_entropy_bits(raw)
        ent_weighted += ent * n
        rows.append(
            {
                "name": f"{role}@L{layer}",
                "params": int(n),
                "unique": int(u.size),
                "entropy_bits": round(ent, 2),
            }
        )

    total_params = sum(r["params"] for r in rows)
    report = {
        "tensors": len(rows),
        "total_params": total_params,
        "dense_MiB": dense_bits / 8 / 1024**2,
        "cross_tensor": {
            "sum_per_tensor_unique": int(sum(per_tensor_unique)),
            "global_unique_routes": len(global_vals),
            "sharing_factor": sum(per_tensor_unique) / max(1, len(global_vals)),
        },
        "lossless_ceilings_vs_dense": {
            "order0_entropy": ent_weighted / total_params / 16,
            "zlib": zlib_bits / dense_bits,
            "lzma": lzma_bits / dense_bits,
            "bitplane_split_zlib": bitplane_zlib_bits / dense_bits,
        },
        "global_min_width_bits_per_weight": math.ceil(math.log2(max(2, len(global_vals)))),
        "note": (
            "route-id map == weights relabeled, so these are hard ceilings for any lossless "
            "route-map codec. sharing_factor ~1 => no cross-tensor route reuse to exploit."
        ),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "routemap_ceiling.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
