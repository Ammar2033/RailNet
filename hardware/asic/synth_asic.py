"""ASIC Synthesis and Elaboration Verification for RailNet Caravel Wrapper.

Validates that wishbone_to_axi.v, caravel_railnet.v, and railnet_top.v
elaborate cleanly without syntax or port mismatches, and computes
pre-layout gate-level area statistics.
"""

import json
from pathlib import Path
from yowasp_yosys import run_yosys

ROOT = Path(__file__).resolve().parents[2]
ASIC_DIR = ROOT / "hardware" / "asic"
BUILD_DIR = ASIC_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)

ys_script = BUILD_DIR / "synth_asic.ys"
stat_json = BUILD_DIR / "caravel_railnet.stat.json"
log_file = BUILD_DIR / "caravel_railnet.log"

wb2axi_rel = (ASIC_DIR / "wishbone_to_axi.v").relative_to(ROOT).as_posix()
sram_rel = (ASIC_DIR / "sky130_sram_macros.v").relative_to(ROOT).as_posix()
caravel_rel = (ASIC_DIR / "caravel_railnet.v").relative_to(ROOT).as_posix()
top_rel = (ROOT / "hardware" / "rtl" / "build" / "railnet_top.v").relative_to(ROOT).as_posix()
dual_clk_rel = (ROOT / "hardware" / "rtl" / "build" / "railnet_dual_clk_top.v").relative_to(ROOT).as_posix()
stat_rel = stat_json.relative_to(ROOT).as_posix()
log_rel = log_file.relative_to(ROOT).as_posix()
ys_rel = ys_script.relative_to(ROOT).as_posix()

commands = (
    f"read_verilog {wb2axi_rel}\n"
    f"read_verilog {sram_rel}\n"
    f"read_verilog {top_rel}\n"
    f"read_verilog {dual_clk_rel}\n"
    f"read_verilog {caravel_rel}\n"
    f"hierarchy -check -top caravel_railnet\n"
    f"proc\n"
    f"opt\n"
    f"memory -nomap\n"
    f"tee -o {stat_rel} stat -json\n"
)

with open(ys_script, "w", encoding="utf-8", newline="\n") as f:
    f.write(commands)

print(f"Running macro-aware ASIC synthesis on caravel_railnet...")
rc = run_yosys(["-q", "-s", ys_rel, "-l", log_rel])
if rc != 0:
    print(f"FAILED (code {rc})! See log at {log_file}")
    with open(log_file, "r", encoding="utf-8") as f:
        print(f.read()[-2000:])
    raise RuntimeError(f"Yosys ASIC elaboration failed with return code {rc}")

with open(stat_json, "r", encoding="utf-8") as f:
    data = json.load(f)

modules = data.get("modules", {})
top_stat = modules.get("\\caravel_railnet") or modules.get("caravel_railnet") or {}
cells = top_stat.get("num_cells_by_type", {})
wire_count = top_stat.get("num_wires", 0)

# Check design totals
design_stat = data.get("design", {})
total_cells = design_stat.get("num_cells", top_stat.get("num_cells", 0))

# Count memory macros preserved
mem_macros = 0
for mod_name, mod_data in modules.items():
    m_cells = mod_data.get("num_cells_by_type", {})
    mem_macros += m_cells.get("$mem_v2", 0) + m_cells.get("$mem", 0)

report = {
    "module": "caravel_railnet",
    "status": "ELABORATION_ONLY",
    "evidence_level": "SYNTHESIZED (GENERIC RTLIL ELABORATION ONLY)",
    "standard_cell_tech_mapped": False,
    "physical_macros_present": False,
    "wires": wire_count,
    "generic_rtlil_cells": total_cells,
    "generic_mem_v2_macros": mem_macros,
    "top_cells": cells,
    "audit_notes": {
        "cell_type_reality": "Cells are Yosys internal RTLIL primitives ($add, $mux, $sdff, $mem_v2), NOT sky130_fd_sc_hd standard cells.",
        "sram_reality": "sky130_sram_macros.v is a behavioral simulation model inside `ifndef SYNTHESIS. No physical OpenRAM macro exists in repository.",
        "missing_tapeout_artifacts": [
            "OpenRAM physical LEF (.lef)",
            "OpenRAM timing Liberty (.lib)",
            "OpenRAM layout mask (.gds)",
            "OpenRAM LVS SPICE (.spice)",
            "Sky130 standard-cell Liberty techmapping (sky130_fd_sc_hd)",
            "Post-route sign-off STA with SPEF parasitics",
            "OpenLane Placement & Clock Tree Synthesis (CTS)",
            "Magic DRC and Netgen LVS tapeout sign-off"
        ]
    }
}

out_report = ROOT / "results" / "asic_synth.json"
out_report.parent.mkdir(exist_ok=True)
with open(out_report, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 65)
print("  RAILNET SKY130 ELABORATION & REALITY AUDIT")
print("=" * 65)
print(f"Top Module:                 caravel_railnet")
print(f"Evidence Level:             SYNTHESIZED (GENERIC RTLIL ONLY)")
print(f"Standard Cell Mapped:       NO (Generic $add, $mux, $mem_v2)")
print(f"Physical OpenRAM Macros:    NO (Behavioral wrapper only)")
print(f"Internal Wires:             {wire_count}")
print(f"Generic RTLIL Cells:        {total_cells}")
print(f"Generic Memory Blocks:      {mem_macros} ($mem_v2 blocks)")
print(f"Sign-off STA Executed:      NO (Theoretical target only)")
print("=" * 65)
print(f"Artifact written to {out_report}")
