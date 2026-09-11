// =============================================================================
// ECP5 PLL Stub for RailNet Dual-Clock Generation
// =============================================================================
// Generates pcie_clk (50-62.5 MHz, host domain) and core_clk (80-100 MHz, core domain)
// from sys_clk 50 MHz onboard oscillator.
//
// For open-source synthesis (Yosys + nextpnr-ecp5):
//   - In simulation, this is a simple clock passthrough (sys_clk -> both).
//   - For P&R, nextpnr will treat this as a placeholder; the LPF
//     FREQUENCY constraints (ecp5_versa.lpf:20) define the timing targets.
//   - For Lattice Diamond/Radiant, replace this stub with an EHXPLLL IP
//     generated via Lattice Clarity Designer (see pll_ecp5_lattice.ipx).
//
// For Lattice ECP5-45F/85F, the real PLL primitive is:
//   EHXPLLL #(.CLKI_DIV(1), .CLKFB_DIV(2), .CLKOP_DIV(4), ...) pll_i (...)
// But we keep this stub portable for Yosys.
//
// Interface:
//   sys_clk  : 50 MHz input (LVDS diff or single-ended)
//   pcie_clk : Host domain clock (50 MHz for FTDI, 62.5 MHz for PCIe)
//   core_clk : Core domain clock (85 MHz default, 100 MHz max)
//   locked   : PLL lock indicator (always 1 in sim, tie to 1'b1 if unused)
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module pll_ecp5 (
    input  wire sys_clk,   // 50 MHz onboard
    output wire pcie_clk,  // 50-62.5 MHz host
    output wire core_clk,  // 85-100 MHz core
    output wire locked
);
    // Simulation: direct passthrough (both clocks = sys_clk)
    // Synthesis: Yosys will keep this as 1 LUT, actual PLL is not inferred.
    // For accurate timing, use Lattice EHXPLLL in Diamond and set
    // FREQUENCY constraints in LPF. This stub is for functional simulation.
`ifdef SIMULATION
    assign pcie_clk = sys_clk;
    assign core_clk = sys_clk;
    assign locked   = 1'b1;
`else
    // For synthesis, infer a simple bypass. nextpnr will use the LPF
    // FREQUENCY constraints to time the design. If you need a real PLL,
    // replace this with:
    //
    //   EHXPLLL #(
    //     .CLKI_DIV(1), .CLKOP_DIV(2), .CLKOS_DIV(2), .CLKOS2_DIV(4),
    //     .FEEDBK_PATH("CLKOP"), .CLKOP_ENABLE("ENABLED")
    //   ) pll_inst (
    //     .CLKI(sys_clk), .CLKOP(core_clk), .CLKOS(pcie_clk),
    //     .CLKFB(sys_clk), .CLKINTFB(), .PHASESEL0(1'b0), .PHASESEL1(1'b0),
    //     .PHASEDIR(1'b0), .PHASESTEP(1'b0), .PHASELOADREG(1'b0),
    //     .STDBY(1'b0), .PLLWAKESYNC(1'b0), .RST(1'b0), .ENCLKOP(1'b1),
    //     .ENCLKOS(1'b1), .LOCK(locked)
    //   );
    //
    // For now, simple assign keeps Yosys/nextpnr happy and timing is
    // constrained via LPF FREQUENCY NET "core_clk" 85.0 MHz.
    assign pcie_clk = sys_clk;
    assign core_clk = sys_clk;
    assign locked   = 1'b1;
`endif

endmodule

`default_nettype wire
