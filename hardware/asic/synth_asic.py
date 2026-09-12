"""Sky130 standard-cell synthesis and honest area reporting for caravel_railnet.

Two modes, and the report always says which one ran:

  * Liberty present -> real ABC technology mapping onto ``sky130_fd_sc_hd``
    cells, with chip area in um^2 taken from the Liberty cell areas.
  * Liberty absent  -> generic RTLIL elaboration only, reported as BLOCKED.
    No cell count is presented as an ASIC area, and nothing is faked.

The Liberty file is third-party and ~12 MB, so it is not vendored (``*.lib`` is
gitignored). Fetch it once:

    mkdir -p hardware/asic/pdk
    curl -L -o hardware/asic/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib \\
      https://raw.githubusercontent.com/efabless/skywater-pdk-libs-sky130_fd_sc_hd/master/timing/sky130_fd_sc_hd__tt_025C_1v80.lib

or point ``SKY130_LIB`` at an existing copy.

Scope, stated plainly:

  * Inferred memories are kept as ``$mem_v2`` (``memory -nomap``) and are NOT
    mapped to standard cells. Mapping them would expand a 256x32 array into
    tens of thousands of flip-flops and report an area that no real chip would
    ever have. The reported area is therefore *logic only*; the memories still
    need real OpenRAM macros, which is Gate 4 and remains open.
  * ``sky130_sram_macros.v`` is not read: no module in this design instantiates
    ``sky130_sram_1kbyte_*`` / ``2kbyte`` / ``4kbyte``. It is a behavioural
    model of macros that were never wired in, so feeding it to synthesis would
    imply an integration that does not exist.
  * Pre-layout only. No placement, no routing, no parasitics, no STA.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from yowasp_yosys import run_yosys

ROOT = Path(__file__).resolve().parents[2]
ASIC_DIR = ROOT / "hardware" / "asic"
RTL_BUILD = ROOT / "hardware" / "rtl" / "build"
BUILD_DIR = ASIC_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)

DEFAULT_LIBERTY = ASIC_DIR / "pdk" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
LIBERTY_URL = (
    "https://raw.githubusercontent.com/efabless/"
    "skywater-pdk-libs-sky130_fd_sc_hd/master/timing/"
    "sky130_fd_sc_hd__tt_025C_1v80.lib"
)

# Which module to synthesise. The committed Gate 3 evidence
# (hardware/asic/gate3_evidence.json) measures `railnet_top`, the accelerator
# core, because that is what RailNet's area claims are about; the Caravel
# wrapper is harness, not RailNet silicon.
TOP = os.environ.get("RAILNET_ASIC_TOP", "railnet_top")

ys_script = BUILD_DIR / "synth_asic.ys"
stat_json = BUILD_DIR / "caravel_railnet.stat.json"
log_file = BUILD_DIR / "caravel_railnet.log"


def resolve_liberty() -> Path | None:
    """Return the Liberty file to map against, or None if unavailable."""
    env = os.environ.get("SKY130_LIB")
    candidate = Path(env) if env else DEFAULT_LIBERTY
    if not candidate.is_file():
        return None
    # A truncated download is worse than a missing one: it would map against a
    # partial cell set and still produce a number.
    head = candidate.open("r", encoding="utf-8", errors="replace").read(4096)
    if "library (" not in head:
        return None
    return candidate


def ensure_rtl_exported() -> list[Path]:
    """Make sure the exported Verilog exists (hardware/rtl/build is gitignored)."""
    wanted = [RTL_BUILD / "railnet_top.v", RTL_BUILD / "railnet_dual_clk_top.v"]
    if all(p.is_file() for p in wanted):
        return wanted
    export = ROOT / "hardware" / "rtl" / "export.py"
    print("[synth_asic] exported Verilog missing; regenerating via hardware/rtl/export.py")
    for args in ([], ["--dual-clock"]):
        subprocess.run([sys.executable, str(export), *args], cwd=str(ROOT), check=True)
    missing = [p for p in wanted if not p.is_file()]
    if missing:
        raise RuntimeError(f"export.py did not produce: {', '.join(str(m) for m in missing)}")
    return wanted


def build_script(liberty: Path | None, sources: list[Path], basenames: bool = False) -> str:
    """Yosys script: elaboration only, or full standard-cell mapping.

    With ``basenames`` the script refers to files by bare name, which is what
    the container path needs: everything is staged flat into one directory.
    """
    if basenames:
        reads = "".join(f"read_verilog {p.name}\n" for p in sources)
        stat_rel = stat_json.name
    else:
        reads = "".join(f"read_verilog {p.relative_to(ROOT).as_posix()}\n" for p in sources)
        stat_rel = stat_json.relative_to(ROOT).as_posix()

    if liberty is None:
        # Elaboration only. This is the old behaviour, and it is not an ASIC area.
        return (
            f"{reads}"
            f"hierarchy -check -top {TOP}\n"
            "proc\n"
            "opt\n"
            "memory -nomap\n"
            f"tee -o {stat_rel} stat -json\n"
        )

    # Must follow the same convention as the reads above: the container stages
    # every input flat into one directory, so an absolute host path is wrong there.
    lib = liberty.name if basenames else liberty.as_posix()
    return (
        f"{reads}"
        f"hierarchy -check -top {TOP}\n"
        "proc\n"
        # Flattened on purpose: without it, Yosys 0.38 re-parses the 12.8 MB
        # Liberty once per module (20+ times observed) and the run crawls.
        # Flattened, ABC is called once and the whole thing takes ~11 s natively.
        "flatten\n"
        "opt_expr\n"
        "opt_clean\n"
        "check\n"
        # Keep inferred memories out of the standard-cell area (see module docstring).
        "memory -nomap\n"
        "opt -full\n"
        "techmap\n"
        "opt -fast\n"
        f"dfflibmap -liberty {lib}\n"
        # Native Yosys 0.38 accepts `-fast` with `-liberty`; the WASM build does not.
        f"abc -fast -liberty {lib}\n"
        "setundef -zero\n"
        "splitnets\n"
        "opt_clean -purge\n"
        f"tee -o {stat_rel} stat -liberty {lib} -json\n"
    )


def run_yosys_in_container(script_text: str, inputs: list[Path]) -> int:
    """Run a Yosys script natively in the OpenLane container.

    Everything is staged into a temp directory and the script refers to files by
    bare name, for two reasons: the repo may live on a drive Docker Desktop will
    not bind-mount (mounting F: fails here with "mkdir /run/desktop/mnt/host/f:
    file exists"), and basenames keep the script identical inside and out.
    Outputs are copied back into hardware/asic/build/.

    Why bother: yowasp-yosys is WebAssembly and cannot finish this mapping.
    Yosys 0.4x there re-parses the 12.8 MB Liberty once per module (20+ times
    observed), and the flattened variant ran 40+ minutes without completing.
    Native Yosys does the same work in about 11 seconds.
    """
    image = os.environ.get("RAILNET_OPENLANE_IMAGE", "efabless/openlane:latest")
    with tempfile.TemporaryDirectory(prefix="railnet-synth-") as tmp:
        stage = Path(tmp)
        for src in inputs:
            shutil.copy2(src, stage / src.name)
        (stage / "synth.ys").write_text(script_text, encoding="utf-8", newline="\n")

        print(f"[synth_asic] native Yosys via {image} (staged in {stage})")
        rc = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{stage}:/work",
                "-w", "/work",
                image,
                "yosys", "-q", "-s", "synth.ys", "-l", "synth.log",
            ],
            env={**os.environ, "MSYS_NO_PATHCONV": "1"},
        ).returncode

        for produced, dest in (
            ("synth.log", log_file),
            (stat_json.name, stat_json),
        ):
            src = stage / produced
            if src.is_file():
                shutil.copy2(src, dest)
    return rc


def parse_chip_area(log_text: str) -> tuple[float | None, dict[str, float]]:
    """Return (top-level area, per-module areas) from a `stat -liberty` log.

    Yosys prints one 'Chip area for module ...' line per module and a final
    'Chip area for top module ...' that accumulates submodules. Without
    flattening, the per-module lines are the useful breakdown.
    """
    per_module: dict[str, float] = {}
    for name, value in re.findall(
        r"Chip area for module '\\?([^']+)':\s*([0-9.]+)", log_text
    ):
        per_module[name] = round(float(value), 3)

    top = re.search(r"Chip area for top module '\\?[^']+':\s*([0-9.]+)", log_text)
    if top:
        return round(float(top.group(1)), 3), per_module
    if per_module:
        return round(max(per_module.values()), 3), per_module
    return None, per_module


def main() -> int:
    liberty = resolve_liberty()
    exported = ensure_rtl_exported()
    by_name = {p.name: p for p in exported}
    if TOP == "railnet_top":
        sources = [by_name["railnet_top.v"]]
    else:
        # caravel_railnet instantiates only wishbone_to_axi and
        # railnet_dual_clk_top; reading railnet_top.v too would just double the
        # work for a module the top never references.
        sources = [
            ASIC_DIR / "wishbone_to_axi.v",
            by_name["railnet_dual_clk_top.v"],
            ASIC_DIR / "caravel_railnet.v",
        ]

    use_docker = os.environ.get("RAILNET_YOSYS_DOCKER") == "1"
    script_text = build_script(liberty, sources, basenames=use_docker)
    ys_script.write_text(script_text, encoding="utf-8", newline="\n")

    mode = "standard-cell tech mapping" if liberty else "generic elaboration (no Liberty)"
    print(f"Running Sky130 ASIC synthesis on {TOP}: {mode}")
    if use_docker:
        inputs = [*sources] + ([liberty] if liberty else [])
        rc = run_yosys_in_container(script_text, inputs)
    else:
        rc = run_yosys(["-q", "-s", ys_script.relative_to(ROOT).as_posix(),
                        "-l", log_file.relative_to(ROOT).as_posix()])
    if rc != 0:
        print(f"FAILED (code {rc})! See log at {log_file}")
        print(log_file.read_text(encoding="utf-8", errors="replace")[-2000:])
        return 1

    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    data = json.loads(stat_json.read_text(encoding="utf-8"))

    modules = data.get("modules", {})
    top_stat = modules.get(f"\\{TOP}") or modules.get(TOP) or {}
    cells = top_stat.get("num_cells_by_type", {})
    wire_count = top_stat.get("num_wires", 0)
    design_stat = data.get("design", {})
    total_cells = design_stat.get("num_cells", top_stat.get("num_cells", 0))

    all_cells: dict[str, int] = {}
    mem_macros = 0
    for mod_data in modules.values():
        for cell_type, n in mod_data.get("num_cells_by_type", {}).items():
            all_cells[cell_type] = all_cells.get(cell_type, 0) + n
            if cell_type in ("$mem_v2", "$mem"):
                mem_macros += n

    std_cells = {k: v for k, v in all_cells.items() if k.startswith("sky130_fd_sc_hd__")}
    std_cell_count = sum(std_cells.values())
    generic_cells = {k: v for k, v in all_cells.items() if k.startswith("$")}
    mapped = liberty is not None and std_cell_count > 0
    chip_area, per_module_area = parse_chip_area(log_text) if mapped else (None, {})
    if mapped:
        # `stat -liberty -json` records area directly. Yosys 0.38 does not always
        # print a "Chip area" line, so the JSON is the authoritative source and
        # the log regex is only a fallback for versions that do print it.
        json_area = top_stat.get("area") or data.get("design", {}).get("area")
        if json_area:
            chip_area = round(float(json_area), 3)
        if not per_module_area:
            per_module_area = {
                name.lstrip("\\"): round(float(md["area"]), 3)
                for name, md in modules.items()
                if md.get("area")
            }

    report = {
        "module": "caravel_railnet",
        "status": "TECH_MAPPED" if mapped else "ELABORATION_ONLY",
        "evidence_level": (
            "SYNTHESIZED (STD-CELL TECH-MAPPED, sky130_fd_sc_hd, pre-layout)"
            if mapped
            else "SYNTHESIZED (GENERIC RTLIL ELABORATION ONLY)"
        ),
        "standard_cell_tech_mapped": mapped,
        "physical_macros_present": False,
        "liberty_file": liberty.as_posix() if liberty else None,
        "liberty_source": LIBERTY_URL if liberty else None,
        "wires": wire_count,
        "generic_rtlil_cells": total_cells,
        "generic_mem_v2_macros": mem_macros,
        "std_cell_count": std_cell_count,
        "logic_area_um2": chip_area,
        "logic_area_um2_by_module": per_module_area,
        "remaining_generic_cells": generic_cells,
        "std_cell_histogram": dict(sorted(std_cells.items(), key=lambda kv: -kv[1])[:20]),
        "top_cells": cells,
        "audit_notes": {
            "area_scope": (
                "Logic only, pre-layout. Inferred memories are kept as $mem_v2 and are "
                "excluded; mapping them to flip-flops would report an area no real chip "
                "would have. Real SRAM macros (OpenRAM) are Gate 4 and still missing."
                if mapped
                else "No area computed: no Liberty file, so no cell mapping was performed."
            ),
            "cell_type_reality": (
                "Cells are sky130_fd_sc_hd standard cells mapped by ABC against the "
                "Liberty file named in liberty_file."
                if mapped
                else "Cells are Yosys internal RTLIL primitives ($add, $mux, $sdff, "
                     "$mem_v2), NOT sky130_fd_sc_hd standard cells."
            ),
            "sram_reality": (
                "sky130_sram_macros.v is a behavioral simulation model inside "
                "`ifndef SYNTHESIS. No physical OpenRAM macro exists in repository. "
                "It is also not instantiated by any module in this design, so it is "
                "excluded from synthesis; hardware/asic/openlane/config.json still "
                "lists it."
            ),
            "not_performed": [
                "Placement, routing, and clock tree synthesis (OpenLane / OpenROAD)",
                "Post-route STA with SPEF parasitics (OpenSTA)",
                "OpenRAM physical macros (.lef / .lib / .gds / .spice)",
                "Magic DRC and Netgen LVS sign-off",
            ],
        },
    }

    out_report = ROOT / "results" / "asic_synth.json"
    out_report.parent.mkdir(exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("  RAILNET SKY130 SYNTHESIS REPORT")
    print("=" * 68)
    print(f"Top module:                 {TOP}")
    print(f"Evidence level:             {report['evidence_level']}")
    print(f"Standard cell mapped:       {'YES' if mapped else 'NO'}")
    if mapped:
        print(f"Liberty:                    {liberty.name}")
        print(f"sky130_fd_sc_hd cells:      {std_cell_count}")
        print(f"Logic area (um^2):          {chip_area}")
        for mod, area in sorted(per_module_area.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {mod:<34} {area:>12.3f}")
        print(f"Unmapped generic cells:     {sum(generic_cells.values())} {list(generic_cells)[:6]}")
    else:
        print("Liberty file:               MISSING -> mapping BLOCKED")
        print(f"  expected at:              {DEFAULT_LIBERTY}")
        print(f"  or set SKY130_LIB; fetch: {LIBERTY_URL}")
        print(f"Generic RTLIL cells:        {total_cells}")
    print(f"Inferred memories ($mem_v2):{mem_macros}  (excluded from area; need OpenRAM)")
    print(f"Physical OpenRAM macros:    NO (Gate 4 open)")
    print(f"Sign-off STA executed:      NO (pre-layout only)")
    print("=" * 68)
    print(f"Artifact written to {out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
