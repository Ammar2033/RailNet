// =============================================================================
// RailNet Low-Cost FPGA + PCIe End-to-End Prototype Hardware Wrapper
// =============================================================================
// Target Platforms:
//   1. QMTech Xilinx Artix-7 XC7A35T/100T PCIe Board (~$70-$90) + LitePCIe IP
//   2. Lattice ECP5-85F Dev Board (~$50-$80) + High-Speed Sync FIFO Bridge
//
// Interface Protocol:
//   - Host CSR: AXI4-Lite (32-bit addr, 32-bit data) -> Register configuration
//   - Host DMA H2C (Host to Card): AXI4-Stream (16-bit/32-bit data) -> Input activations
//   - Host DMA C2H (Card to Host): AXI4-Stream (32-bit data) -> Output activations
//   - Core Interrupt: IRQ line routed to PCIe MSI / MSI-X vector
// =============================================================================

`timescale 1ns / 1ps

module railnet_pcie_wrapper #(
    parameter integer NUM_TILES = 4,
    parameter integer RAILS     = 16,
    parameter integer DATA_W    = 32,
    parameter integer ACT_W     = 16,
    parameter integer ADDR_W    = 12
)(
    // -------------------------------------------------------------------------
    // PCIe / Host Clock & Reset
    // -------------------------------------------------------------------------
    input  wire                  pcie_clk,        // 62.5 MHz / 125 MHz PCIe AXI clock
    input  wire                  pcie_rst_n,      // Active-low PCIe fundamental reset (PERST#)
    input  wire                  core_clk,        // Accelerator core clock (PLL generated)
    input  wire                  core_rst_n,      // Accelerator synchronous reset

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (PCIe BAR0 / CSR Config)
    // -------------------------------------------------------------------------
    input  wire [ADDR_W-1:0]     s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output wire                  s_axi_awready,
    input  wire [31:0]           s_axi_wdata,
    input  wire [3:0]            s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output wire                  s_axi_wready,
    output wire [1:0]            s_axi_bresp,
    output wire                  s_axi_bvalid,
    input  wire                  s_axi_bready,
    input  wire [ADDR_W-1:0]     s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output wire                  s_axi_arready,
    output wire [31:0]           s_axi_rdata,
    output wire [1:0]            s_axi_rresp,
    output wire                  s_axi_rvalid,
    input  wire                  s_axi_rready,

    // -------------------------------------------------------------------------
    // AXI4-Stream Host-to-Card DMA (H2C) - Input Activations
    // -------------------------------------------------------------------------
    input  wire [ACT_W-1:0]      s_axis_dma_tdata,
    input  wire                  s_axis_dma_tvalid,
    output wire                  s_axis_dma_tready,
    input  wire                  s_axis_dma_tlast,

    // -------------------------------------------------------------------------
    // AXI4-Stream Card-to-Host DMA (C2H) - Output Results
    // -------------------------------------------------------------------------
    output wire [DATA_W-1:0]     m_axis_dma_tdata,
    output wire                  m_axis_dma_tvalid,
    input  wire                  m_axis_dma_tready,
    output wire                  m_axis_dma_tlast,

    // -------------------------------------------------------------------------
    // Interrupt / Status Flags
    // -------------------------------------------------------------------------
    output wire                  irq_out,         // Level-sensitive or pulsed MSI interrupt
    output wire [3:0]            debug_leds       // Hardware status diagnostic LEDs
);

    // Synchronize resets (active-high for Amaranth modules)
    wire host_reset = ~pcie_rst_n;
    wire core_reset = ~core_rst_n;

    // -------------------------------------------------------------------------
    // Instantiate Dual-Clock RailNet Hardware Top
    // -------------------------------------------------------------------------
    // Bridges host domain (PCIe) and core domain (Accelerator) via internal
    // Gray-coded AsyncFIFOs and CDC synchronizers.
    railnet_top u_railnet_core (
        .clk              (core_clk),
        .rst              (core_reset),

        // AXI4-Lite
        .s_axi_awaddr     (s_axi_awaddr),
        .s_axi_awvalid    (s_axi_awvalid),
        .s_axi_awready    (s_axi_awready),
        .s_axi_wdata      (s_axi_wdata),
        .s_axi_wstrb      (s_axi_wstrb),
        .s_axi_wvalid     (s_axi_wvalid),
        .s_axi_wready     (s_axi_wready),
        .s_axi_bresp      (s_axi_bresp),
        .s_axi_bvalid     (s_axi_bvalid),
        .s_axi_bready     (s_axi_bready),
        .s_axi_araddr     (s_axi_araddr),
        .s_axi_arvalid    (s_axi_arvalid),
        .s_axi_arready    (s_axi_arready),
        .s_axi_rdata      (s_axi_rdata),
        .s_axi_rresp      (s_axi_rresp),
        .s_axi_rvalid     (s_axi_rvalid),
        .s_axi_rready     (s_axi_rready),
        .irq              (irq_out),

        // AXI4-Stream Input (DMA H2C)
        .s_axis_tdata     (s_axis_dma_tdata),
        .s_axis_tvalid    (s_axis_dma_tvalid),
        .s_axis_tready    (s_axis_dma_tready),
        .s_axis_tlast     (s_axis_dma_tlast),

        // AXI4-Stream Output (DMA C2H)
        .m_axis_tdata     (m_axis_dma_tdata),
        .m_axis_tvalid    (m_axis_dma_tvalid),
        .m_axis_tready    (m_axis_dma_tready),
        .m_axis_tlast     (m_axis_dma_tlast),

        // Dedicated in-band programming interface (tied off to CSR control)
        .prog_tile_idx    ({$clog2(NUM_TILES){1'b0}}),
        .prog_route_en    (1'b0),
        .prog_route_addr  (16'd0),
        .prog_route_data  (16'd0),
        .prog_cb_en       (1'b0),
        .prog_cb_addr     (16'd0),
        .prog_cb_data     (32'd0),
        .prog_rail_en     (1'b0),
        .prog_rail_addr   (8'd0),
        .prog_rail_data   (8'd0)
    );

    // Diagnostic status indicators for physical board testing
    assign debug_leds[0] = s_axis_dma_tvalid & s_axis_dma_tready; // Input streaming active
    assign debug_leds[1] = m_axis_dma_tvalid & m_axis_dma_tready; // Output streaming active
    assign debug_leds[2] = irq_out;                               // Frame completion IRQ
    assign debug_leds[3] = pcie_rst_n;                            // PCIe link active (alive LED)

endmodule
