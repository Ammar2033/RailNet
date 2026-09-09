"""Export RailNet accelerator to synthesizable Verilog-2001.

Usage:
    python hardware/rtl/export.py [--tiles 4] [--out railnet_top.v]
    python hardware/rtl/export.py --dual-clock [--tiles 4] [--out railnet_dual_clk_top.v]
"""

import argparse
from pathlib import Path
import sys

from amaranth.back import verilog

from hardware.rtl.top import RailNetTop


def export_verilog(num_tiles: int = 4, rails: int = 32, out_path: str = "hardware/rtl/build/railnet_top.v"):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=64, route_depth=512)

    ports = [
        # AXI4-Lite
        dut.s_axi_awaddr,
        dut.s_axi_awvalid,
        dut.s_axi_awready,
        dut.s_axi_wdata,
        dut.s_axi_wstrb,
        dut.s_axi_wvalid,
        dut.s_axi_wready,
        dut.s_axi_bresp,
        dut.s_axi_bvalid,
        dut.s_axi_bready,
        dut.s_axi_araddr,
        dut.s_axi_arvalid,
        dut.s_axi_arready,
        dut.s_axi_rdata,
        dut.s_axi_rresp,
        dut.s_axi_rvalid,
        dut.s_axi_rready,
        dut.irq,
        # AXI4-Stream In
        dut.s_axis_tdata,
        dut.s_axis_tvalid,
        dut.s_axis_tready,
        dut.s_axis_tlast,
        # AXI4-Stream Out
        dut.m_axis_tdata,
        dut.m_axis_tvalid,
        dut.m_axis_tready,
        dut.m_axis_tlast,
        # Programming
        dut.prog_tile_idx,
        dut.prog_route_en,
        dut.prog_route_addr,
        dut.prog_route_data,
        dut.prog_cb_en,
        dut.prog_cb_addr,
        dut.prog_cb_data,
        dut.prog_rail_en,
        dut.prog_rail_addr,
        dut.prog_rail_data,
    ]

    print(f"Converting RailNetTop ({num_tiles} tiles, {rails} rails) to RTLIL...")
    from amaranth.back import rtlil
    from yowasp_yosys import run_yosys

    il_text = rtlil.convert(dut, name="railnet_top", ports=ports)
    root = Path(__file__).resolve().parents[2]
    il_file = out_file.with_suffix(".il")
    with open(il_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(il_text)

    ys_file = out_file.with_suffix(".ys")
    il_rel = il_file.resolve().relative_to(root).as_posix()
    out_rel = out_file.resolve().relative_to(root).as_posix()
    ys_rel = ys_file.resolve().relative_to(root).as_posix()

    ys_script = (
        f"read_rtlil {il_rel}\n"
        f"hierarchy -top railnet_top\n"
        f"proc\n"
        f"opt\n"
        f"write_verilog -noattr {out_rel}\n"
    )
    with open(ys_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(ys_script)

    print("Running yowasp-yosys to generate clean Verilog-2001...")
    run_yosys(["-q", "-s", ys_rel])

    if out_file.exists():
        v_code = out_file.read_text(encoding="utf-8")
        print(f"Successfully generated synthesizable Verilog: {out_file.resolve()} ({len(v_code):,} chars, {v_code.count(chr(10))} lines)")
        return out_file
    raise RuntimeError(f"Failed to generate {out_file}")


def export_dual_clock_verilog(
    num_tiles: int = 4,
    rails: int = 32,
    out_path: str = "hardware/rtl/build/railnet_dual_clk_top.v",
):
    """Export RailNetDualClockTop (host_clk + core_clk) to Verilog."""
    from amaranth import ClockSignal, ResetSignal
    from hardware.rtl.cdc import RailNetDualClockTop

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dut = RailNetDualClockTop(
        num_tiles=num_tiles, rails=rails,
        codebook_depth=64, route_depth=512, fifo_depth=32,
    )

    ports = [
        # Clock and Reset (exported as top-level ports)
        ClockSignal("host"), ResetSignal("host"),
        ClockSignal("core"), ResetSignal("core"),
        # AXI4-Lite
        dut.s_axi_awaddr, dut.s_axi_awvalid, dut.s_axi_awready,
        dut.s_axi_wdata, dut.s_axi_wstrb, dut.s_axi_wvalid, dut.s_axi_wready,
        dut.s_axi_bresp, dut.s_axi_bvalid, dut.s_axi_bready,
        dut.s_axi_araddr, dut.s_axi_arvalid, dut.s_axi_arready,
        dut.s_axi_rdata, dut.s_axi_rresp, dut.s_axi_rvalid, dut.s_axi_rready,
        dut.irq,
        # AXI4-Stream
        dut.s_axis_tdata, dut.s_axis_tvalid, dut.s_axis_tready, dut.s_axis_tlast,
        dut.m_axis_tdata, dut.m_axis_tvalid, dut.m_axis_tready, dut.m_axis_tlast,
        # Programming
        dut.prog_tile_idx, dut.prog_route_en, dut.prog_route_addr, dut.prog_route_data,
        dut.prog_cb_en, dut.prog_cb_addr, dut.prog_cb_data,
        dut.prog_rail_en, dut.prog_rail_addr, dut.prog_rail_data,
    ]

    top_name = "railnet_dual_clk_top"
    print(f"Converting RailNetDualClockTop ({num_tiles} tiles, {rails} rails, dual-clock) to RTLIL...")
    from amaranth.back import rtlil
    from yowasp_yosys import run_yosys

    il_text = rtlil.convert(dut, name=top_name, ports=ports)
    root = Path(__file__).resolve().parents[2]
    il_file = out_file.with_suffix(".il")
    with open(il_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(il_text)

    ys_file = out_file.with_suffix(".ys")
    il_rel = il_file.resolve().relative_to(root).as_posix()
    out_rel = out_file.resolve().relative_to(root).as_posix()
    ys_rel = ys_file.resolve().relative_to(root).as_posix()

    ys_script = (
        f"read_rtlil {il_rel}\n"
        f"hierarchy -top {top_name}\n"
        f"proc\n"
        f"opt\n"
        f"write_verilog -noattr {out_rel}\n"
    )
    with open(ys_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(ys_script)

    print("Running yowasp-yosys to generate clean Verilog-2001 (dual-clock)...")
    run_yosys(["-q", "-s", ys_rel])

    if out_file.exists():
        v_code = out_file.read_text(encoding="utf-8")
        print(f"Successfully generated dual-clock Verilog: {out_file.resolve()} ({len(v_code):,} chars, {v_code.count(chr(10))} lines)")
        return out_file
    raise RuntimeError(f"Failed to generate {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Export RailNetTop to Verilog")
    parser.add_argument("--tiles", type=int, default=4, help="Number of compute tiles (default: 4)")
    parser.add_argument("--rails", type=int, default=32, help="Number of shared rails (default: 32)")
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    parser.add_argument("--dual-clock", action="store_true", help="Export dual-clock CDC version")
    args = parser.parse_args()

    if args.dual_clock:
        out = args.out or "hardware/rtl/build/railnet_dual_clk_top.v"
        export_dual_clock_verilog(num_tiles=args.tiles, rails=args.rails, out_path=out)
    else:
        out = args.out or "hardware/rtl/build/railnet_top.v"
        export_verilog(num_tiles=args.tiles, rails=args.rails, out_path=out)


if __name__ == "__main__":
    main()

