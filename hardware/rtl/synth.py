"""Synthesize the RTL tiles (yosys synth_xilinx) and compare resources.

    PYTHONPATH=. python hardware/rtl/synth.py

Emits hardware/rtl/build/*.il, runs yosys, writes results/rtl_synth.json and a
table. The number that matters: stage-A gather fabric vs one dense MAC.
"""

import json
import re
from pathlib import Path

from amaranth.back import rtlil
from yowasp_yosys import run_yosys

from hardware.rtl.tiles import ACC_W, ACT_W, DenseInner, StageA, StageABram, StageB

RAILS = 96
BUILD = Path(__file__).resolve().parent / "build"
ROOT = Path(__file__).resolve().parents[2]


def _stat(name: str) -> dict:
    il = BUILD / f"{name}.il"
    stat = BUILD / f"{name}.stat.json"
    ys = BUILD / f"{name}.ys"
    ys.write_text(
        f"read_rtlil {il}\nhierarchy -top {name}\nsynth_xilinx -flatten\ntee -o {stat} stat -json\n"
    )
    run_yosys(["-q", "-s", str(ys), "-l", str(BUILD / f"{name}.log")])
    d = json.loads(stat.read_text())
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
        "stagea_reg": StageA(RAILS),
        "stagea_bram": StageABram(RAILS),
        "stageb": StageB(),
    }
    report = {"params": {"rails": RAILS, "act_w": ACT_W, "acc_w": ACC_W}, "tiles": {}}
    for name, mod in tiles.items():
        (BUILD / f"{name}.il").write_text(rtlil.convert(mod, name=name))
        cells = _stat(name)
        report["tiles"][name] = {"summary": _summ(cells), "cells": cells}

    d = report["tiles"]["dense"]["summary"]
    for name in ("stagea_reg", "stagea_bram"):
        s = report["tiles"][name]["summary"]
        report["tiles"][name]["vs_dense_lut_equiv"] = round(
            (s["LUT"] + s["FF"] * 0.5 + s["DSP"] * 100 + s["BRAM"] * 50)
            / max(1, d["LUT"] + d["FF"] * 0.5 + d["DSP"] * 100 + d["BRAM"] * 50),
            1,
        )

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rtl_synth.json").write_text(json.dumps(report, indent=2))

    print(f"{'tile':14s} {'DSP':>4} {'BRAM':>5} {'FF':>6} {'LUT':>6} {'CARRY4':>7} {'MUXF':>5}")
    for name, t in report["tiles"].items():
        s = t["summary"]
        extra = f"  ~{t['vs_dense_lut_equiv']}x dense" if "vs_dense_lut_equiv" in t else ""
        print(
            f"{name:14s} {s['DSP']:>4} {s['BRAM']:>5} {s['FF']:>6} {s['LUT']:>6} "
            f"{s['CARRY4']:>7} {s['MUXF']:>5}{extra}"
        )
    print(f"\n-> {ROOT / 'results' / 'rtl_synth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
