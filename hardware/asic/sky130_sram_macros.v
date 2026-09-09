// ===========================================================================
// sky130_sram_macros.v
//
// OpenRAM SRAM Macro Behavioral Models for SkyWater 130nm (sky130A)
// Compatible with OpenLane, OpenROAD, and Caravel harness.
//
// Provides exact pin-compatible blackbox and simulation models for:
// 1. sky130_sram_1kbyte_1rw1r_32x256_8  (1KB: 256 words x 32-bit, 1RW1R)
// 2. sky130_sram_2kbyte_1rw1r_32x512_8  (2KB: 512 words x 32-bit, 1RW1R)
// 3. sky130_sram_4kbyte_1rw1r_32x1024_8 (4KB: 1024 words x 32-bit, 1RW1R)
// ===========================================================================

`default_nettype none

// ---------------------------------------------------------------------------
// 1. 1KB OpenRAM Macro (256 words x 32 bits, 1RW1R)
// ---------------------------------------------------------------------------
module sky130_sram_1kbyte_1rw1r_32x256_8 (
`ifdef USE_POWER_PINS
    inout vccd1,
    inout vssd1,
`endif
    // Port 0: Read/Write (RW)
    input  wire        clk0,
    input  wire        csb0,       // Active low chip select
    input  wire        web0,       // Active low write enable
    input  wire [3:0]  wmask0,     // Byte write mask
    input  wire [7:0]  addr0,      // 8-bit address (256 words)
    input  wire [31:0] din0,
    output reg  [31:0] dout0,

    // Port 1: Read-only (R)
    input  wire        clk1,
    input  wire        csb1,       // Active low chip select
    input  wire [7:0]  addr1,      // 8-bit address (256 words)
    output reg  [31:0] dout1
);

`ifndef SYNTHESIS
    reg [31:0] mem [0:255];

    // Port 0: Synchronous Read/Write
    always @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                if (wmask0[0]) mem[addr0][7:0]   <= din0[7:0];
                if (wmask0[1]) mem[addr0][15:8]  <= din0[15:8];
                if (wmask0[2]) mem[addr0][23:16] <= din0[23:16];
                if (wmask0[3]) mem[addr0][31:24] <= din0[31:24];
            end
            dout0 <= mem[addr0];
        end
    end

    // Port 1: Synchronous Read
    always @(posedge clk1) begin
        if (!csb1) begin
            dout1 <= mem[addr1];
        end
    end
`endif

endmodule


// ---------------------------------------------------------------------------
// 2. 2KB OpenRAM Macro (512 words x 32 bits, 1RW1R)
// ---------------------------------------------------------------------------
module sky130_sram_2kbyte_1rw1r_32x512_8 (
`ifdef USE_POWER_PINS
    inout vccd1,
    inout vssd1,
`endif
    // Port 0: Read/Write (RW)
    input  wire        clk0,
    input  wire        csb0,
    input  wire        web0,
    input  wire [3:0]  wmask0,
    input  wire [8:0]  addr0,      // 9-bit address (512 words)
    input  wire [31:0] din0,
    output reg  [31:0] dout0,

    // Port 1: Read-only (R)
    input  wire        clk1,
    input  wire        csb1,
    input  wire [8:0]  addr1,
    output reg  [31:0] dout1
);

`ifndef SYNTHESIS
    reg [31:0] mem [0:511];

    always @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                if (wmask0[0]) mem[addr0][7:0]   <= din0[7:0];
                if (wmask0[1]) mem[addr0][15:8]  <= din0[15:8];
                if (wmask0[2]) mem[addr0][23:16] <= din0[23:16];
                if (wmask0[3]) mem[addr0][31:24] <= din0[31:24];
            end
            dout0 <= mem[addr0];
        end
    end

    always @(posedge clk1) begin
        if (!csb1) begin
            dout1 <= mem[addr1];
        end
    end
`endif

endmodule


// ---------------------------------------------------------------------------
// 3. 4KB OpenRAM Macro (1024 words x 32 bits, 1RW1R)
// ---------------------------------------------------------------------------
module sky130_sram_4kbyte_1rw1r_32x1024_8 (
`ifdef USE_POWER_PINS
    inout vccd1,
    inout vssd1,
`endif
    // Port 0: Read/Write (RW)
    input  wire        clk0,
    input  wire        csb0,
    input  wire        web0,
    input  wire [3:0]  wmask0,
    input  wire [9:0]  addr0,      // 10-bit address (1024 words)
    input  wire [31:0] din0,
    output reg  [31:0] dout0,

    // Port 1: Read-only (R)
    input  wire        clk1,
    input  wire        csb1,
    input  wire [9:0]  addr1,
    output reg  [31:0] dout1
);

`ifndef SYNTHESIS
    reg [31:0] mem [0:1023];

    always @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                if (wmask0[0]) mem[addr0][7:0]   <= din0[7:0];
                if (wmask0[1]) mem[addr0][15:8]  <= din0[15:8];
                if (wmask0[2]) mem[addr0][23:16] <= din0[23:16];
                if (wmask0[3]) mem[addr0][31:24] <= din0[31:24];
            end
            dout0 <= mem[addr0];
        end
    end

    always @(posedge clk1) begin
        if (!csb1) begin
            dout1 <= mem[addr1];
        end
    end
`endif

endmodule
