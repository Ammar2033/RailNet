/*
 * Wishbone B4 Slave to AXI4-Lite Master Bridge
 *
 * Converts 32-bit Wishbone transactions from Caravel SoC core
 * to AXI4-Lite CSR transactions for RailNet Top.
 */

`default_nettype none

module wishbone_to_axi (
    input  wire        clk,
    input  wire        rst,

    // Wishbone Slave
    input  wire        wbs_cyc_i,
    input  wire        wbs_stb_i,
    input  wire        wbs_we_i,
    input  wire [3:0]  wbs_sel_i,
    input  wire [31:0] wbs_adr_i,
    input  wire [31:0] wbs_dat_i,
    output reg         wbs_ack_o,
    output reg  [31:0] wbs_dat_o,

    // AXI4-Lite Master
    output reg  [11:0] m_axi_awaddr,
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,

    output reg  [31:0] m_axi_wdata,
    output reg  [3:0]  m_axi_wstrb,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,

    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,

    output reg  [11:0] m_axi_araddr,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,

    input  wire [31:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready
);

    localparam STATE_IDLE      = 3'd0;
    localparam STATE_WRITE     = 3'd1;
    localparam STATE_WRITE_RSP = 3'd2;
    localparam STATE_READ_ADDR = 3'd3;
    localparam STATE_READ_DATA = 3'd4;
    localparam STATE_ACK       = 3'd5;

    reg [2:0] state;
    reg aw_done;
    reg w_done;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state         <= STATE_IDLE;
            wbs_ack_o     <= 1'b0;
            wbs_dat_o     <= 32'h00000000;
            m_axi_awaddr  <= 12'h000;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata   <= 32'h00000000;
            m_axi_wstrb   <= 4'h0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
            m_axi_araddr  <= 12'h000;
            m_axi_arvalid <= 1'b0;
            m_axi_rready  <= 1'b0;
            aw_done       <= 1'b0;
            w_done        <= 1'b0;
        end else begin
            wbs_ack_o <= 1'b0;

            case (state)
                STATE_IDLE: begin
                    if (wbs_cyc_i && wbs_stb_i && !wbs_ack_o) begin
                        if (wbs_we_i) begin
                            // Write transaction
                            m_axi_awaddr  <= wbs_adr_i[11:0];
                            m_axi_awvalid <= 1'b1;
                            m_axi_wdata   <= wbs_dat_i;
                            m_axi_wstrb   <= wbs_sel_i;
                            m_axi_wvalid  <= 1'b1;
                            aw_done       <= 1'b0;
                            w_done        <= 1'b0;
                            state         <= STATE_WRITE;
                        end else begin
                            // Read transaction
                            m_axi_araddr  <= wbs_adr_i[11:0];
                            m_axi_arvalid <= 1'b1;
                            state         <= STATE_READ_ADDR;
                        end
                    end
                end

                STATE_WRITE: begin
                    if (m_axi_awready && m_axi_awvalid) begin
                        m_axi_awvalid <= 1'b0;
                        aw_done       <= 1'b1;
                    end
                    if (m_axi_wready && m_axi_wvalid) begin
                        m_axi_wvalid  <= 1'b0;
                        w_done        <= 1'b1;
                    end

                    if ((aw_done || (m_axi_awready && m_axi_awvalid)) &&
                        (w_done  || (m_axi_wready && m_axi_wvalid))) begin
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b1;
                        state         <= STATE_WRITE_RSP;
                    end
                end

                STATE_WRITE_RSP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;
                        wbs_ack_o    <= 1'b1;
                        state        <= STATE_ACK;
                    end
                end

                STATE_READ_ADDR: begin
                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        state         <= STATE_READ_DATA;
                    end
                end

                STATE_READ_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        m_axi_rready <= 1'b0;
                        wbs_dat_o    <= m_axi_rdata;
                        wbs_ack_o    <= 1'b1;
                        state        <= STATE_ACK;
                    end
                end

                STATE_ACK: begin
                    wbs_ack_o <= 1'b0;
                    state     <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
