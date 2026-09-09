"""Synthesize the RTL tiles (yosys synth_xilinx) and compare resources.

    python hardware/rtl/synth.py

Emits hardware/rtl/build/*.il, runs yosys synth_xilinx, writes results/rtl_synth.json
and prints a resource comparison table.
Focus: Stage-A gather fabric and full BF16 tile vs Dense MAC baselines.
"""

import json
from pathlib import Path
import re

from amaranth.back import rtlil
from yowasp_yosys import run_yosys

from hardware.rtl.bf16_tile import RailNetFullTile, StageABf16Bram, StageBBf16
from hardware.rtl.int8_tile import RailNetInt8Tile, StageAInt8Bram, StageBInt8
from hardware.rtl.tiles import ACC_W, ACT_W, DenseInner, StageA, StageABram, StageB
from hardware.rtl.top import RailNetTop

RAILS = 96
BUILD = Path(__file__).resolve().parent / "build"
ROOT = Path(__file__).resolve().parents[2]


def _stat(name: str) -> dict:
    il = BUILD / f"{name}.il"
    stat = BUILD / f"{name}.stat.json"
    ys = BUILD / f"{name}.ys"
    log = BUILD / f"{name}.log"

    il_rel = il.relative_to(ROOT).as_posix()
    stat_rel = stat.relative_to(ROOT).as_posix()
    ys_rel = ys.relative_to(ROOT).as_posix()
    log_rel = log.relative_to(ROOT).as_posix()

    ys_content = (
        f"read_rtlil {il_rel}\n"
        f"hierarchy -top {name}\n"
        f"synth_xilinx -flatten\n"
        f"tee -o {stat_rel} stat -json\n"
    )
    with open(ys, "w", encoding="utf-8", newline="\n") as f:
        f.write(ys_content)

    run_yosys(["-q", "-s", ys_rel, "-l", log_rel])
    d = json.loads(stat.read_text(encoding="utf-8"))
    top = d["modules"].get(name) or next(iter(d["modules"].values()))
    return {k: v for k, v in top["num_cells_by_type"].items() if v}


def _summ(cells: dict) -> dict:
    lut = sum(v for k, v in cells.items() if re.fullmatch(r"LUT[1-6]", k))
    ff = sum(v for k, v in cells.items() if k.startswith("FD"))
    dsp = sum(v for k, v in cells.items() if k.startswith("DSP"))
    bram = sum(v for k, v in cells.items() if k.startswith("RAMB"))
    carry = sum(v for k, v in cells.items() if k.startswith("CARRY"))
    muxf = sum(v for k, v in cells.items() if k.startswith("MUXF"))
    return {"DSP": dsp, "BRAM": bram, "FF": ff, "LUT": lut, "CARRY4": carry, "MUXF": muxf}


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    tiles = {
        "dense": DenseInner(),
        "stagea_bram": StageABram(RAILS),
        "stagea_bf16_bram": StageABf16Bram(RAILS),
        "stageb_bf16": StageBBf16(RAILS),
        "full_bf16_tile": RailNetFullTile(RAILS, codebook_depth=64),
        "stagea_int8_bram": StageAInt8Bram(32),
        "stageb_int8": StageBInt8(32),
        "full_int8_tile": RailNetInt8Tile(32, codebook_depth=64),
        "railnet_top_2x2": RailNetTop(num_tiles=4, rails=32, codebook_depth=64, route_depth=256),
    }
    report = {"params": {"rails": RAILS, "act_w": ACT_W, "acc_w": ACC_W}, "tiles": {}}
    for name, mod in tiles.items():
        print(f"Synthesizing {name}...")
        il_text = rtlil.convert(mod, name=name)
        with open(BUILD / f"{name}.il", "w", encoding="utf-8", newline="\n") as f:
            f.write(il_text)
        cells = _stat(name)
        report["tiles"][name] = {"summary": _summ(cells), "cells": cells}

    d = report["tiles"]["dense"]["summary"]
    for name in report["tiles"]:
        if name == "dense":
            continue
        s = report["tiles"][name]["summary"]
        report["tiles"][name]["vs_dense_lut_equiv"] = round(
            (s["LUT"] + s["FF"] * 0.5 + s["DSP"] * 100 + s["BRAM"] * 50)
            / max(1, d["LUT"] + d["FF"] * 0.5 + d["DSP"] * 100 + d["BRAM"] * 50),
            1,
        )

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rtl_synth.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"{'tile':18s} {'DSP':>4} {'BRAM':>5} {'FF':>6} {'LUT':>6} {'CARRY4':>7} {'MUXF':>5}")
    print("=" * 65)
    for name, t in report["tiles"].items():
        s = t["summary"]
        extra = f"  ~{t['vs_dense_lut_equiv']}x dense" if "vs_dense_lut_equiv" in t else ""
        print(
            f"{name:18s} {s['DSP']:>4} {s['BRAM']:>5} {s['FF']:>6} {s['LUT']:>6} "
            f"{s['CARRY4']:>7} {s['MUXF']:>5}{extra}"
        )
    print("=" * 65)
    print(f"\n-> {ROOT / 'results' / 'rtl_synth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
