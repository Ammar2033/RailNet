// =============================================================================
// Artix-7 Clocking Wizard Stub for RailNet
// =============================================================================
// Generates pcie_clk (62.5 MHz, LitePCIe user_clk) and core_clk (100 MHz)
// from sys_clk 50 MHz onboard oscillator via MMCM.
//
// For Vivado:
//   - Replace this stub with a real Clocking Wizard IP (clk_wiz_0.xci)
//     generated via IP Catalog: Clocking Wizard -> MMCM, 50 MHz in,
//     CLK_OUT1 = 62.5 MHz (pcie_clk), CLK_OUT2 = 100 MHz (core_clk).
//   - The XDC `qmtech_artix7_pcie.xdc:75` already has
//     create_clock for sys_clk/pcie_clk/core_clk and
//     set_clock_groups -asynchronous for CDC.
//   - Vivado will use the XCI's MMCME2_ADV primitive for accurate
//     timing and will generate the required clock buffers (BUFG).
//
// For Yosys (yowasp-yosys) open-source synthesis:
//   - This stub is a simple passthrough, Yosys will synthesize it as
//     2 LUTs. The XDC frequency constraints still apply for timing
//     estimation. For accurate 7-series tech mapping, Yosys synth_xilinx
//     will map the MMCM stub to LUTs, but the real MMCM will be
//     inferred only in Vivado.
//
// Interface:
//   sys_clk  : 50 MHz input (LVCMOS33, pin M21 on QMTech)
//   pcie_clk : 62.5 MHz host (LitePCIe) or 125 MHz (XDMA) - default 62.5
//   core_clk : 100 MHz core (configurable 80-100 MHz)
//   locked   : MMCM lock indicator
//   resetn   : Active-low reset (from PERST# or pushbutton)
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module clk_wiz_artix7 (
    input  wire sys_clk,   // 50 MHz
    input  wire resetn,    // Active-low
    output wire pcie_clk,  // 62.5 MHz
    output wire core_clk,  // 100 MHz
    output wire locked
);
    // Simulation / Yosys stub: passthrough
    // In Vivado, this will be replaced by clk_wiz_0.xci (MMCME2_ADV)
`ifdef SIMULATION
    assign pcie_clk = sys_clk;
    assign core_clk = sys_clk;
    assign locked   = 1'b1;
`else
    // Yosys / next steps: simple assign (see note above)
    // To use a real MMCM in Vivado, uncomment the instance below
    // and comment out the assigns. Ensure you generate the IP via:
    //   Tools -> IP Catalog -> Clocking Wizard -> 50 MHz in, 62.5 & 100 out
    //
    //   MMCME2_ADV #(
    //     .BANDWIDTH("OPTIMIZED"), .CLKFBOUT_MULT_F(20.0), .CLKIN1_PERIOD(20.0),
    //     .CLKOUT0_DIVIDE_F(16.0), // 62.5 MHz
    //     .CLKOUT1_DIVIDE(10),     // 100 MHz
    //     .DIVCLK_DIVIDE(1)
    //   ) mmcm_inst (
    //     .CLKIN1(sys_clk), .CLKFBIN(sys_clk), .CLKFBOUT(),
    //     .CLKOUT0(pcie_clk), .CLKOUT1(core_clk), .LOCKED(locked),
    //     .RST(~resetn), .PWRDWN(1'b0)
    //   );
    //
    assign pcie_clk = sys_clk;
    assign core_clk = sys_clk;
    assign locked   = resetn; // tie to resetn for sim
`endif

endmodule

`default_nettype wire
