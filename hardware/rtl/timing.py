"""Place-and-Route (P&R) timing analysis on ECP5 using Yosys and nextpnr.

    python hardware/rtl/timing.py

Workflow:
1. Converts Amaranth component to RTLIL (.il).
2. Synthesizes netlist for ECP5 using Yosys (`synth_ecp5 -top <name> -json <name>.json`).
3. Runs Place-and-Route with `nextpnr-ecp5` targeting LFE5U-85F speed grade 8 at 100 MHz.
4. Parses achieved Fmax (MHz), critical path delay (ns), slack, and resource utilization.
5. Writes results to results/rtl_timing.json and prints a formatted summary table.
"""

import json
from pathlib import Path
import re

from amaranth.back import rtlil
from yowasp_nextpnr_ecp5 import run_nextpnr_ecp5
from yowasp_yosys import run_yosys

from hardware.rtl.bf16_tile import RailNetFullTile, StageABf16Bram, StageBBf16
from hardware.rtl.tiles import DenseInner, StageABram

BUILD = Path(__file__).resolve().parent / "build"
ROOT = Path(__file__).resolve().parents[2]


def _synth_ecp5(name: str) -> Path:
    """Synthesize module to ECP5 JSON netlist using Yosys."""
    il = BUILD / f"{name}.il"
    json_out = BUILD / f"{name}_ecp5.json"
    ys = BUILD / f"{name}_ecp5.ys"
    log = BUILD / f"{name}_ecp5_synth.log"

    il_rel = il.relative_to(ROOT).as_posix()
    json_rel = json_out.relative_to(ROOT).as_posix()
    ys_rel = ys.relative_to(ROOT).as_posix()
    log_rel = log.relative_to(ROOT).as_posix()

    ys_content = (
        f"read_rtlil {il_rel}\n"
        f"hierarchy -top {name}\n"
        f"synth_ecp5 -top {name} -json {json_rel}\n"
    )
    with open(ys, "w", encoding="utf-8", newline="\n") as f:
        f.write(ys_content)

    run_yosys(["-q", "-s", ys_rel, "-l", log_rel])
    return json_out


def _pnr_ecp5(name: str, json_path: Path, target_freq_mhz: float = 100.0) -> dict:
    """Run nextpnr-ecp5 on synthesized JSON netlist and extract timing metrics."""
    log_out = BUILD / f"{name}_pnr.log"
    json_rel = json_path.relative_to(ROOT).as_posix()
    log_rel = log_out.relative_to(ROOT).as_posix()

    # Arguments for nextpnr-ecp5
    args = [
        "--85k",
        "--package", "CABGA381",
        "--speed", "8",
        "--freq", str(target_freq_mhz),
        "--json", json_rel,
        "-l", log_rel,
        "-q",
    ]

    try:
        run_nextpnr_ecp5(args)
    except SystemExit:
        pass

    log_text = log_out.read_text(encoding="utf-8", errors="replace")

    # Extract Max frequency (Fmax)
    # Example log: "Max frequency for clock '$glbnet$clk': 78.43 MHz (FAIL at 100.00 MHz)"
    fmax = None
    clk_match = re.search(r"Max frequency for clock\s+['\$\w]+:\s*([\d\.]+)\s*MHz", log_text)
    if clk_match:
        fmax = float(clk_match.group(1))

    # Extract critical path delay
    crit_delay = None
    delay_match = re.search(r"critical path delay:\s*([\d\.]+)\s*ns", log_text, re.IGNORECASE)
    if delay_match:
        crit_delay = float(delay_match.group(1))
    elif fmax and fmax > 0:
        crit_delay = round(1000.0 / fmax, 3)

    # Extract device utilization
    lut_match = re.search(r"TRELLIS_COMB:\s*(\d+)/", log_text)
    ff_match = re.search(r"TRELLIS_FF:\s*(\d+)/", log_text)
    dsp_match = re.search(r"MULT18X18D:\s*(\d+)/", log_text)

    lut_count = int(lut_match.group(1)) if lut_match else 0
    ff_count = int(ff_match.group(1)) if ff_match else 0
    dsp_count = int(dsp_match.group(1)) if dsp_match else 0

    return {
        "target_freq_mhz": target_freq_mhz,
        "achieved_fmax_mhz": fmax,
        "critical_path_ns": crit_delay,
        "ecp5_lut": lut_count,
        "ecp5_ff": ff_count,
        "ecp5_dsp": dsp_count,
    }


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    tiles = {
        "dense": DenseInner(),
        "stagea_bram": StageABram(96),
        "stagea_bf16_bram": StageABf16Bram(96),
        "stageb_bf16": StageBBf16(96),
        "full_bf16_tile": RailNetFullTile(96, codebook_depth=64),
    }

    report = {"device": "LFE5U-85F-8CABGA381", "target_freq_mhz": 100.0, "tiles": {}}

    print("=================================================================")
    print(" Running ECP5 Place-and-Route (P&R) Timing Analysis with nextpnr")
    print(" Device: Lattice ECP5 LFE5U-85F (Speed Grade: 8) Target: 100 MHz")
    print("=================================================================\n")

    for name, mod in tiles.items():
        print(f"[{name}] Generating RTLIL...")
        il_text = rtlil.convert(mod, name=name)
        with open(BUILD / f"{name}.il", "w", encoding="utf-8", newline="\n") as f:
            f.write(il_text)

        print(f"[{name}] Yosys synth_ecp5...")
        json_path = _synth_ecp5(name)

        print(f"[{name}] nextpnr-ecp5 place & route...")
        res = _pnr_ecp5(name, json_path, target_freq_mhz=100.0)
        report["tiles"][name] = res
        print(f"[{name}] Fmax: {res['achieved_fmax_mhz']} MHz | Critical delay: {res['critical_path_ns']} ns\n")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rtl_timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'tile':18s} {'Fmax (MHz)':>11} {'Crit Path (ns)':>15} {'LUT':>6} {'FF':>6} {'DSP':>5}")
    print("=" * 70)
    for name, r in report["tiles"].items():
        fmax_str = f"{r['achieved_fmax_mhz']:.2f}" if r['achieved_fmax_mhz'] else "N/A"
        crit_str = f"{r['critical_path_ns']:.3f}" if r['critical_path_ns'] else "N/A"
        print(
            f"{name:18s} {fmax_str:>11} {crit_str:>15} {r['ecp5_lut']:>6} {r['ecp5_ff']:>6} {r['ecp5_dsp']:>5}"
        )
    print("=" * 70)
    print(f"\n-> {ROOT / 'results' / 'rtl_timing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
