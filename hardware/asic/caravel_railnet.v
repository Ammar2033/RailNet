/*
 *-------------------------------------------------------------
 *
 * caravel_railnet.v
 *
 * Efabless Caravel SoC User Project Wrapper for RailNet AI Accelerator
 * Compatible with SkyWater 130nm PDK and OpenLane flow.
 *
 * Provides:
 *   1. Wishbone B4 CSR interface via wishbone_to_axi bridge
 *   2. Logic Analyzer (LA) activation streaming and result capture
 *   3. Hardware IRQ generation and GPIO status telemetry
 *   4. Dual-clock CDC: host_clk (wb_clk_i) + core_clk (user_clock2 PLL)
 *
 * Clock Domains:
 *   host_clk  = wb_clk_i      (Caravel Wishbone bus clock, 50-125 MHz)
 *   core_clk  = user_clock2    (Caravel PLL output, 200-500 MHz)
 *
 *-------------------------------------------------------------
 */

`default_nettype none

`ifndef MPRJC_IO_PADS
`define MPRJC_IO_PADS 38
`endif

module caravel_railnet (
`ifdef USE_POWER_PINS
    inout wire vccd1, // 1.8V Core digital power
    inout wire vssd1, // Core digital ground
    inout wire vccd2,
    inout wire vssd2,
    inout wire vdda1,
    inout wire vssa1,
    inout wire vdda2,
    inout wire vssa2,
`endif

    // Wishbone Slave
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,
    input  wire        wbs_stb_i,
    input  wire        wbs_cyc_i,
    input  wire        wbs_we_i,
    input  wire [3:0]  wbs_sel_i,
    input  wire [31:0] wbs_dat_i,
    input  wire [31:0] wbs_adr_i,
    output wire        wbs_ack_o,
    output wire [31:0] wbs_dat_o,

    // Caravel PLL output → core_clk for compute pipeline
    input  wire        user_clock2,

    // Logic Analyzer Signals
    input  wire [127:0] la_data_in,
    output wire [127:0] la_data_out,
    input  wire [127:0] la_oenb,

    // IOs
    input  wire [`MPRJC_IO_PADS-1:0] io_in,
    output wire [`MPRJC_IO_PADS-1:0] io_out,
    output wire [`MPRJC_IO_PADS-1:0] io_oeb,

    // IRQ
    output wire [2:0] user_irq
);

    // Internal AXI-Lite Bus Wires
    wire [11:0] axi_awaddr;
    wire        axi_awvalid;
    wire        axi_awready;
    wire [31:0] axi_wdata;
    wire [3:0]  axi_wstrb;
    wire        axi_wvalid;
    wire        axi_wready;
    wire [1:0]  axi_bresp;
    wire        axi_bvalid;
    wire        axi_bready;
    wire [11:0] axi_araddr;
    wire        axi_arvalid;
    wire        axi_arready;
    wire [31:0] axi_rdata;
    wire [1:0]  axi_rresp;
    wire        axi_rvalid;
    wire        axi_rready;

    // Internal Streaming & IRQ Wires
    wire [31:0] s_axis_tdata;
    wire        s_axis_tvalid;
    wire        s_axis_tready;
    wire        s_axis_tlast;

    wire [31:0] m_axis_tdata;
    wire        m_axis_tvalid;
    wire        m_axis_tready;
    wire        m_axis_tlast;

    wire        railnet_irq;

    // 1. Wishbone to AXI-Lite Bridge
    wishbone_to_axi u_wb2axi (
        .clk(wb_clk_i),
        .rst(wb_rst_i),

        .wbs_cyc_i(wbs_cyc_i),
        .wbs_stb_i(wbs_stb_i),
        .wbs_we_i(wbs_we_i),
        .wbs_sel_i(wbs_sel_i),
        .wbs_adr_i(wbs_adr_i),
        .wbs_dat_i(wbs_dat_i),
        .wbs_ack_o(wbs_ack_o),
        .wbs_dat_o(wbs_dat_o),

        .m_axi_awaddr(axi_awaddr),
        .m_axi_awvalid(axi_awvalid),
        .m_axi_awready(axi_awready),
        .m_axi_wdata(axi_wdata),
        .m_axi_wstrb(axi_wstrb),
        .m_axi_wvalid(axi_wvalid),
        .m_axi_wready(axi_wready),
        .m_axi_bresp(axi_bresp),
        .m_axi_bvalid(axi_bvalid),
        .m_axi_bready(axi_bready),
        .m_axi_araddr(axi_araddr),
        .m_axi_arvalid(axi_arvalid),
        .m_axi_arready(axi_arready),
        .m_axi_rdata(axi_rdata),
        .m_axi_rresp(axi_rresp),
        .m_axi_rvalid(axi_rvalid),
        .m_axi_rready(axi_rready)
    );

    // 2. Map Stream Channels to Logic Analyzer / GPIO
    assign s_axis_tdata  = la_data_in[31:0];
    assign s_axis_tvalid = la_data_in[32];
    assign s_axis_tlast  = la_data_in[33];
    assign m_axis_tready = la_data_in[34];

    assign la_data_out[31:0]   = m_axis_tdata;
    assign la_data_out[32]     = m_axis_tvalid;
    assign la_data_out[33]     = m_axis_tlast;
    assign la_data_out[34]     = s_axis_tready;
    assign la_data_out[35]     = railnet_irq;
    assign la_data_out[127:36] = 92'd0;

    // 3. Connect GPIO Telemetry
    assign io_out[0] = railnet_irq;
    assign io_out[1] = s_axis_tready;
    assign io_out[2] = m_axis_tvalid;
    assign io_out[3] = m_axis_tlast;
    assign io_out[`MPRJC_IO_PADS-1:4] = {(`MPRJC_IO_PADS-4){1'b0}};
    assign io_oeb = { {(`MPRJC_IO_PADS-4){1'b1}}, 4'b0000 };

    // 4. Connect User IRQ
    assign user_irq = {2'b00, railnet_irq};

    // 5. Dual-Clock Core: host_clk = wb_clk_i, core_clk = user_clock2 (PLL)
    //    When user_clock2 is not driven by PLL, tie to wb_clk_i for
    //    single-clock backward compatibility.
    wire core_clk_sel;
    assign core_clk_sel = user_clock2;

    // Reset Synchronizer for core_clk domain:
    // Asynchronous assertion, 2-stage synchronous deassertion to prevent recovery violations
    reg [1:0] core_rst_sync;
    always @(posedge core_clk_sel or posedge wb_rst_i) begin
        if (wb_rst_i)
            core_rst_sync <= 2'b11;
        else
            core_rst_sync <= {core_rst_sync[0], 1'b0};
    end
    wire core_rst_synced = core_rst_sync[1];

    railnet_dual_clk_top u_railnet (
        .host_clk(wb_clk_i),
        .host_rst(wb_rst_i),
        .core_clk(core_clk_sel),
        .core_rst(core_rst_synced),

        .s_axi_awaddr(axi_awaddr),
        .s_axi_awvalid(axi_awvalid),
        .s_axi_awready(axi_awready),
        .s_axi_wdata(axi_wdata),
        .s_axi_wstrb(axi_wstrb),
        .s_axi_wvalid(axi_wvalid),
        .s_axi_wready(axi_wready),
        .s_axi_bresp(axi_bresp),
        .s_axi_bvalid(axi_bvalid),
        .s_axi_bready(axi_bready),
        .s_axi_araddr(axi_araddr),
        .s_axi_arvalid(axi_arvalid),
        .s_axi_arready(axi_arready),
        .s_axi_rdata(axi_rdata),
        .s_axi_rresp(axi_rresp),
        .s_axi_rvalid(axi_rvalid),
        .s_axi_rready(axi_rready),

        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),

        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),

        .prog_tile_idx(la_data_in[37:36]),
        .prog_route_en(la_data_in[38]),
        .prog_route_addr(la_data_in[47:39]),
        .prog_route_data(la_data_in[63:48]),
        .prog_cb_en(la_data_in[64]),
        .prog_cb_addr(la_data_in[80:65]),
        .prog_cb_data(la_data_in[107:81]),
        .prog_rail_en(la_data_in[108]),
        .prog_rail_addr(la_data_in[113:109]),
        .prog_rail_data(la_data_in[121:114]),

        .irq(railnet_irq)
    );

endmodule
