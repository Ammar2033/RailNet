"""Is structured routing (ADR 0001, option A') viable? — decisive probe.

The current route-id map costs 16 bits/weight because each weight names one of
~3-5k global routes. Structured routing would instead store, per block, the
small set of rails that block actually uses, and per weight only a local index
+ signs.

This measures — from the already-compiled Gemma3 1B artifacts, no retraining —
how many *distinct rails* each block of weights touches. If blocks are
rail-local (few distinct rails), A' shrinks the map a lot, losslessly. If not,
the ~12-bit entropy floor is real and the path is B (dense NVM) or C.

    python research/structured_routing_probe.py --compiled compiled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = [("row", None), ("16x16", 16), ("32x32", 32), ("64x64", 64)]


def _rail_of_route_id(routes: dict) -> dict[int, frozenset]:
    return {int(b): frozenset(int(r) for r, _s in terms) for b, terms in routes.items()}


def _block_view(a: np.ndarray, bs: int | None):
    m, n = a.shape
    if bs is None:  # rows
        return a.reshape(m, 1, n)
    pm, pn = (-m) % bs, (-n) % bs
    g = np.pad(a, ((0, pm), (0, pn)), constant_values=a.flat[0])
    return g.reshape(g.shape[0] // bs, bs, g.shape[1] // bs, bs).swapaxes(1, 2).reshape(-1, bs, bs)


def probe_tensor(artifact: dict, route_ids: np.ndarray) -> dict:
    rail_sets = _rail_of_route_id(artifact["routes"])
    max_terms = int(artifact["max_terms"])
    n_terms = np.vectorize(lambda b: len(rail_sets[int(b)]))(route_ids)

    out = {"shape": list(route_ids.shape), "avg_terms": float(n_terms.mean())}
    dense_bits = route_ids.size * 16

    for label, bs in BLOCKS:
        # distinct rails touched by a block = union of route rails over its weights
        blocks = _block_view(route_ids, bs)
        nb, h, w = blocks.shape
        bw = h * w
        distinct = np.empty(nb, dtype=np.int32)
        for i in range(nb):
            u = np.unique(blocks[i])
            s: set[int] = set()
            for b in u:
                s |= rail_sets[int(b)]
            distinct[i] = len(s)

        # structured cost: per block, store its rail set (distinct*7 bits) +
        # per weight max_terms local indices into it (ceil(log2(distinct)) each) + signs
        loc_bits = np.maximum(1, np.ceil(np.log2(np.maximum(2, distinct))))
        per_block_hdr = distinct * 7
        per_weight = max_terms * (loc_bits + 1)  # +1 sign bit per term
        struct_bits = int((per_block_hdr + bw * per_weight).sum())

        out[label] = {
            "blocks": int(nb),
            "distinct_rails_mean": float(distinct.mean()),
            "distinct_rails_p95": float(np.percentile(distinct, 95)),
            "structured_bits_ratio": struct_bits / dense_bits,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", default=str(ROOT / "compiled"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.compiled)
    manifest = json.loads((out / "manifest.json").read_text())
    entries = [e for e in manifest["tensors"].values() if e.get("status") == "PASS"]
    if args.limit:
        entries = entries[: args.limit]

    agg: dict[str, list] = {label: [] for label, _ in BLOCKS}
    rows = {}
    for i, e in enumerate(entries):
        art = json.loads((out / e["artifact"]).read_text())
        ids = np.load(out / e["route_map"])
        r = probe_tensor(art, ids)
        rows[f"{e['role']}@L{e['layer']}"] = r
        for label, _ in BLOCKS:
            agg[label].append(r[label]["structured_bits_ratio"])
        print(f"[{i + 1}/{len(entries)}] {e['role']}@L{e['layer']}", flush=True)

    summary = {
        "tensors": len(entries),
        "structured_ratio_mean": {k: float(np.mean(v)) for k, v in agg.items()},
        "structured_ratio_best_blocksize": min(
            ((k, float(np.mean(v))) for k, v in agg.items()), key=lambda kv: kv[1]
        ),
        "verdict_hint": (
            "ratio << 1  => A' viable, chip-storage path opens. "
            "ratio ~ 1   => 12-bit entropy floor is real -> path B (dense NVM) or C."
        ),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "structured_routing_probe.json").write_text(
        json.dumps({"summary": summary, "per_tensor": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
