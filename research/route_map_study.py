"""Full route-map compression study over a compiled RailNet directory.

    python research/route_map_study.py --compiled compiled

Runs railnet.analysis.route_compression over every PASS tensor, splits the
result by role (attention vs MLP), and writes results/route_map_study.json plus
a Markdown summary to hardware/research/routing_storage.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from railnet.analysis.route_compression import benchmark_route_map

ROOT = Path(__file__).resolve().parent.parent
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _agg(rows: list[dict]) -> dict:
    tot: dict[str, int] = defaultdict(int)
    dense = 0
    for r in rows:
        dense += r["_dense"]
        for k, v in r.items():
            if not k.startswith("_"):
                tot[k] += v
    dense = dense or 1
    return {
        "dense_MiB": dense / 8 / 1024**2,
        "schemes": dict(
            sorted(
                ((k, {"bits": v, "ratio": v / dense}) for k, v in tot.items()),
                key=lambda kv: kv[1]["bits"],
            )
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", default=str(ROOT / "compiled"))
    args = ap.parse_args()

    out = Path(args.compiled)
    manifest = json.loads((out / "manifest.json").read_text())
    entries = [e for e in manifest["tensors"].values() if e.get("status") == "PASS"]

    by_group: dict[str, list[dict]] = {"all": [], "attention": [], "mlp": []}
    for i, e in enumerate(entries):
        ids = np.load(out / e["route_map"])
        bench = benchmark_route_map(ids, tuple(e["shape"]))
        row = {k: v["bits"] for k, v in bench.items() if not k.startswith("_")}
        row["_dense"] = int(ids.size * 16)
        row["_name"] = f"{e['role']}@L{e['layer']}"
        by_group["all"].append(row)
        by_group["attention" if e["role"] in ATTN else "mlp"].append(row)
        print(f"[{i + 1}/{len(entries)}] {row['_name']}", flush=True)

    report = {
        "compiled_dir": str(out),
        "tensors": len(entries),
        "verdict": manifest.get("verdict"),
        "groups": {g: _agg(rows) for g, rows in by_group.items() if rows},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "route_map_study.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report)
    print(json.dumps({g: v["schemes"] for g, v in report["groups"].items()}, indent=2))
    return 0


_READING = (
    "## Reading\n\n"
    "- The route-id map carries ~`log2(unique_routes)` bits of real entropy per element"
    " (~12 for Gemma3 1B linears) and little spatial structure, so per-block palettes and RLE"
    " are *worse* than dense.\n"
    "- Best lossless is generic entropy coding at ~0.8x dense — consistent with the honest"
    " ~1.23x in `docs/MEMORY.md`. There is no large storage win hiding here.\n"
    "- Implication for hardware: the RailNet case rests on the shared-compute fabric and"
    " wiring, not on smaller weight storage. A hierarchical routing fabric has to justify"
    " itself on area/power/latency, not bit count.\n"
)


def _write_markdown(report: dict) -> None:
    head = (
        f"Study of **{report['tensors']} compiled tensors** "
        f"(`{report['compiled_dir']}`, verdict `{report['verdict']}`), "
        "lossless schemes only. `ratio` is vs dense BF16; **> 1 means larger than dense**."
    )
    lines = ["# Routing storage — measured", "", head, ""]
    for g, data in report["groups"].items():
        lines += [
            f"## {g}  ({data['dense_MiB']:.1f} MiB dense)",
            "",
            "| scheme | ratio vs dense |",
            "|---|---|",
        ]
        lines += [f"| {name} | {s['ratio']:.3f} |" for name, s in data["schemes"].items()]
        lines.append("")
    lines.append(_READING)
    (ROOT / "hardware" / "research" / "routing_storage.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
