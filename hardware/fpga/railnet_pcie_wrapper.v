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
//
// Architecture (Gate 2 Hardened):
//   - Dual-clock CDC via RailNetDualClockTop (hardware/rtl/cdc.py:337)
//   - host_clk (pcie_clk 62.5/125 MHz) <-> core_clk (80-100 MHz) are
//     asynchronous. All crossings use Gray-coded AsyncFIFO (47-bit) and
//     FFSynchronizer/PulseSynchronizer - see hardware/rtl/cdc.py:14.
//   - Programming via in-band CSR (REG_PROG_ADDR/DATA/CTRL) - no external
//     prog_* pins needed for PCIe bring-up. Direct prog_* tied off.
//   - Single-clock fallback: define `SINGLE_CLOCK` to instantiate railnet_top.
// =============================================================================

`timescale 1ns / 1ps

module railnet_pcie_wrapper #(
    parameter integer NUM_TILES = 4,
    parameter integer RAILS     = 32,   // must match RailNetDualClockTop RAILS (default 32)
    parameter integer DATA_W    = 32,
    parameter integer ACT_W     = 16,
    parameter integer ADDR_W    = 12
)(
    // -------------------------------------------------------------------------
    // PCIe / Host Clock & Reset  (Host Domain)
    // -------------------------------------------------------------------------
    input  wire                  pcie_clk,        // 62.5 MHz / 125 MHz PCIe AXI clock (host domain)
    input  wire                  pcie_rst_n,      // Active-low PCIe fundamental reset (PERST#)
    input  wire                  core_clk,        // Accelerator core clock (PLL generated, 80-100 MHz)
    input  wire                  core_rst_n,      // Accelerator synchronous reset (active-low)

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (PCIe BAR0 / CSR Config) - Host Domain
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
    // AXI4-Stream Host-to-Card DMA (H2C) - Input Activations (Host Domain in,
    // crossed to Core Domain via AsyncFIFO inside RailNetDualClockTop)
    // -------------------------------------------------------------------------
    input  wire [ACT_W-1:0]      s_axis_dma_tdata,
    input  wire                  s_axis_dma_tvalid,
    output wire                  s_axis_dma_tready,
    input  wire                  s_axis_dma_tlast,

    // -------------------------------------------------------------------------
    // AXI4-Stream Card-to-Host DMA (C2H) - Output Results (Core Domain out,
    // crossed to Host Domain via AsyncFIFO)
    // -------------------------------------------------------------------------
    output wire [DATA_W-1:0]     m_axis_dma_tdata,
    output wire                  m_axis_dma_tvalid,
    input  wire                  m_axis_dma_tready,
    output wire                  m_axis_dma_tlast,

    // -------------------------------------------------------------------------
    // Interrupt / Status Flags  (Host Domain)
    // -------------------------------------------------------------------------
    output wire                  irq_out,         // Level-sensitive MSI interrupt (host domain sync)
    output wire [3:0]            debug_leds       // Hardware status diagnostic LEDs
);

    // -------------------------------------------------------------------------
    // Clock / Reset Synchronisation (active-high for Amaranth CDC module)
    // -------------------------------------------------------------------------
    wire host_rst = ~pcie_rst_n;  // host domain reset (async assert, sync deassert inside FIFOs)
    wire core_rst = ~core_rst_n;  // core domain reset

    // Sign-extend narrow ACT_W activations to 32-bit core datapath.
    // RailNetTop/CDC expects signed 32-bit; upper bits are sign-extended from ACT_W-1.
    // E.g. ACT_W=16: 0xFF10 (-240) -> 0xFFFF_FF10 ; 0x0010 (16) -> 0x0000_0010
    wire [31:0] s_axis_tdata32;
    // Use generate-style replication that is constant-folded (ACT_W is parameter)
    // For synthesis, (32-ACT_W) must be constant; with ACT_W=16 -> 16.
    assign s_axis_tdata32 = {{(32-ACT_W){s_axis_dma_tdata[ACT_W-1]}}, s_axis_dma_tdata};

    // -------------------------------------------------------------------------
    // Dedicated Prog Interface - Tied off to in-band CSR control
    // (Gate 1 in-band programming via REG_PROG_ADDR 0x18 / REG_PROG_DATA 0x1C / REG_PROG_CTRL 0x20)
    // Widths sized for default RailNetDualClockTop: NUM_TILES=4 => 2b, RAILS=32 => 5b,
    // route_depth=512 => 9b, codebook_depth up to 2048 => 11b (but prog port is 16b)
    // For non-default builds, regenerate railnet_dual_clk_top.v via:
    //   python -m hardware.rtl.export --top railnet_dual_clk_top
    // -------------------------------------------------------------------------
    wire [1:0]  prog_tile_idx_const  = 2'b0;   // $clog2(NUM_TILES)
    wire        prog_route_en_const  = 1'b0;
    wire [8:0]  prog_route_addr_const= 9'd0;   // max(1,(route_depth-1).bit_length())
    wire [15:0] prog_route_data_const=16'd0;
    wire        prog_cb_en_const     = 1'b0;
    wire [15:0] prog_cb_addr_const   = 16'd0;
    wire [26:0] prog_cb_data_const   = 27'd0;
    wire        prog_rail_en_const   = 1'b0;
    wire [4:0]  prog_rail_addr_const = 5'd0;   // max(1,(RAILS-1).bit_length())
    wire [7:0]  prog_rail_data_const = 8'd0;

    // -------------------------------------------------------------------------
    // Instantiate Dual-Clock RailNet Core
    // Host domain (pcie_clk) handles AXI-Lite CSR, AXI-Stream DMA I/O, IRQ.
    // Core domain (core_clk) runs Grid/Broadcaster/Gather/FSM.
    // All crossings are inside RailNetDualClockTop via AsyncFIFO (Gray-coded).
    // -------------------------------------------------------------------------
`ifndef SINGLE_CLOCK
    // Preferred: Dual-clock CDC-hardened top (Gate 2 requirement)
    railnet_dual_clk_top u_railnet_core (
        .host_clk       (pcie_clk),
        .host_rst       (host_rst),
        .core_clk       (core_clk),
        .core_rst       (core_rst),

        // AXI4-Lite - Host Domain
        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),
        .s_axi_araddr   (s_axi_araddr),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),
        .irq            (irq_out),

        // AXI4-Stream Input (Host -> Core CDC)
        .s_axis_tdata   (s_axis_tdata32),
        .s_axis_tvalid  (s_axis_dma_tvalid),
        .s_axis_tready  (s_axis_dma_tready),
        .s_axis_tlast   (s_axis_dma_tlast),

        // AXI4-Stream Output (Core -> Host CDC)
        .m_axis_tdata   (m_axis_dma_tdata),
        .m_axis_tvalid  (m_axis_dma_tvalid),
        .m_axis_tready  (m_axis_dma_tready),
        .m_axis_tlast   (m_axis_dma_tlast),

        // Direct prog interface (tied off - use CSR in-band)
        .prog_tile_idx  (prog_tile_idx_const),
        .prog_route_en  (prog_route_en_const),
        .prog_route_addr(prog_route_addr_const),
        .prog_route_data(prog_route_data_const),
        .prog_cb_en     (prog_cb_en_const),
        .prog_cb_addr   (prog_cb_addr_const),
        .prog_cb_data   (prog_cb_data_const),
        .prog_rail_en   (prog_rail_en_const),
        .prog_rail_addr (prog_rail_addr_const),
        .prog_rail_data (prog_rail_data_const)
    );
`else
    // Fallback: Single-clock RailNetTop (for sim or ECP5 without PCIe SerDes)
    // pcie_clk is ignored; everything runs on core_clk.
    wire _unused_pcie = pcie_clk | host_rst;
    railnet_top u_railnet_core (
        .clk              (core_clk),
        .rst              (core_rst),

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

        // AXI4-Stream Input
        .s_axis_tdata     (s_axis_tdata32),
        .s_axis_tvalid    (s_axis_dma_tvalid),
        .s_axis_tready    (s_axis_dma_tready),
        .s_axis_tlast     (s_axis_dma_tlast),

        // AXI4-Stream Output
        .m_axis_tdata     (m_axis_dma_tdata),
        .m_axis_tvalid    (m_axis_dma_tvalid),
        .m_axis_tready    (m_axis_dma_tready),
        .m_axis_tlast     (m_axis_dma_tlast),

        // Direct prog (tied off)
        .prog_tile_idx    ({$clog2(NUM_TILES){1'b0}}),
        .prog_route_en    (1'b0),
        .prog_route_addr  (9'd0),
        .prog_route_data  (16'd0),
        .prog_cb_en       (1'b0),
        .prog_cb_addr     (16'd0),
        .prog_cb_data     (27'd0),
        .prog_rail_en     (1'b0),
        .prog_rail_addr   (5'd0),
        .prog_rail_data   (8'd0)
    );
`endif

    // -------------------------------------------------------------------------
    // Diagnostic Status Indicators for Physical Board Bring-up
    // -------------------------------------------------------------------------
    // LED0: Input DMA handshake active (host streaming)
    // LED1: Output DMA handshake active (card streaming)
    // LED2: Frame completion IRQ (level)
    // LED3: PCIe link alive (PERST# deasserted)
    assign debug_leds[0] = s_axis_dma_tvalid & s_axis_dma_tready;
    assign debug_leds[1] = m_axis_dma_tvalid & m_axis_dma_tready;
    assign debug_leds[2] = irq_out;
    assign debug_leds[3] = pcie_rst_n;

    // CDC timing exception: host and core are asynchronous
    // Real constraint in XDC: set_clock_groups -asynchronous -group [get_clocks pcie_clk] -group [get_clocks core_clk]

endmodule
