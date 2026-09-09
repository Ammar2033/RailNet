"""AXI4-Lite and AXI4-Stream Interface Components for RailNet.

Provides:
- AxiLiteCsr: Standard 32-bit AXI4-Lite Slave for memory-mapped Control & Status Registers.
- AxiStreamIn: AXI4-Stream Slave receiver for activation streaming.
- AxiStreamOut: AXI4-Stream Master transmitter for computed neuron results.
"""

from amaranth import Cat, Const, Module, Mux, Signal, unsigned
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

ADDR_W = 12
DATA_W = 32

# Register offsets (byte-addressed)
REG_CTRL = 0x00
REG_STATUS = 0x04
REG_IN_FEATURES = 0x08
REG_OUT_FEATURES = 0x0C
REG_TILE_MASK = 0x10
REG_CYCLE_COUNT = 0x14
REG_PROG_ADDR = 0x18
REG_PROG_DATA = 0x1C
REG_PROG_CTRL = 0x20


class AxiLiteCsr(wiring.Component):
    """32-bit AXI4-Lite Slave for Control and Status Registers.

    Ports:
        AXI-Lite Slave Bus:
            awaddr, awvalid, awready
            wdata, wstrb, wvalid, wready
            bresp, bvalid, bready
            araddr, arvalid, arready
            rdata, rresp, rvalid, rready
        Internal Register Interface:
            ctrl_start: Pulsed 1 when host issues start.
            ctrl_soft_reset: Reset command from host.
            ctrl_accumulate: If 1, multi-pass mode: preserve Stage-A BRAM (do not flush).
            status_busy: In-progress flag from core.
            status_done: Finished flag from core.
            in_features: Number of inputs per token.
            out_features: Total output neurons.
            tile_mask: Active tile enable bitmask.
            cycle_count: Cycle performance counter.
            prog_tile_idx, prog_mem_type, prog_addr, prog_data, prog_strobe: In-band programming.
    """

    def __init__(self, addr_w: int = ADDR_W, data_w: int = DATA_W, num_tiles: int = 4):
        self.addr_w = addr_w
        self.data_w = data_w
        self.num_tiles = num_tiles
        tidx_w = max(1, (num_tiles - 1).bit_length())

        super().__init__(
            {
                # AXI-Lite Write Address Channel
                "awaddr": In(addr_w),
                "awvalid": In(1),
                "awready": Out(1),
                # AXI-Lite Write Data Channel
                "wdata": In(data_w),
                "wstrb": In(4),
                "wvalid": In(1),
                "wready": Out(1),
                # AXI-Lite Write Response Channel
                "bresp": Out(2),
                "bvalid": Out(1),
                "bready": In(1),
                # AXI-Lite Read Address Channel
                "araddr": In(addr_w),
                "arvalid": In(1),
                "arready": Out(1),
                # AXI-Lite Read Data Channel
                "rdata": Out(data_w),
                "rresp": Out(2),
                "rvalid": Out(1),
                "rready": In(1),
                # Internal Core Interface
                "ctrl_start": Out(1),
                "ctrl_soft_reset": Out(1),
                "ctrl_accumulate": Out(1),
                "status_busy": In(1),
                "status_done": In(1),
                "in_features": Out(16),
                "out_features": Out(16),
                "tile_mask": Out(num_tiles),
                "cycle_count": In(32),
                "irq": Out(1),
                # Ping-Pong Double-Buffering Interface
                "active_bank": Out(1),
                "prog_bank": Out(1),
                "bank_ready": Out(1),
                "auto_swap_en": Out(1),
                "hw_bank_swap": In(1),
                # Programming Interface
                "prog_tile_idx": Out(tidx_w),
                "prog_mem_type": Out(2),
                "prog_addr": Out(16),
                "prog_data": Out(32),
                "prog_strobe": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        tidx_w = max(1, (self.num_tiles - 1).bit_length())

        # Internal storage registers
        reg_in_feat = Signal(16, reset=128)
        reg_out_feat = Signal(16, reset=self.num_tiles)
        reg_tile_mask = Signal(self.num_tiles, reset=(1 << self.num_tiles) - 1)
        reg_ctrl_start = Signal(1, reset=0)
        reg_ctrl_soft_reset = Signal(1, reset=0)
        reg_ctrl_accumulate = Signal(1, reset=0)
        reg_ctrl_ie = Signal(1, reset=0)

        # Ping-pong bank control registers
        reg_active_bank = Signal(1, reset=0)
        reg_bank_ready = Signal(1, reset=0)
        reg_auto_swap_en = Signal(1, reset=0)

        reg_prog_tile_idx = Signal(tidx_w, reset=0)
        reg_prog_mem_type = Signal(2, reset=0)
        reg_prog_addr = Signal(16, reset=0)
        reg_prog_data = Signal(32, reset=0)
        reg_prog_strobe = Signal(1, reset=0)

        # Drive outputs to core
        m.d.comb += [
            self.in_features.eq(reg_in_feat),
            self.out_features.eq(reg_out_feat),
            self.tile_mask.eq(reg_tile_mask),
            self.ctrl_start.eq(reg_ctrl_start),
            self.ctrl_soft_reset.eq(reg_ctrl_soft_reset),
            self.ctrl_accumulate.eq(reg_ctrl_accumulate),
            self.irq.eq(self.status_done & reg_ctrl_ie),
            self.bresp.eq(0),  # OKAY
            self.rresp.eq(0),  # OKAY
            self.active_bank.eq(reg_active_bank),
            self.prog_bank.eq(Mux(reg_auto_swap_en, ~reg_active_bank, reg_active_bank)),
            self.bank_ready.eq(reg_bank_ready),
            self.auto_swap_en.eq(reg_auto_swap_en),
            self.prog_tile_idx.eq(reg_prog_tile_idx),
            self.prog_mem_type.eq(reg_prog_mem_type),
            self.prog_addr.eq(reg_prog_addr),
            self.prog_data.eq(reg_prog_data),
            self.prog_strobe.eq(reg_prog_strobe),
        ]

        # Auto-clear pulses in next cycle if set
        with m.If(reg_ctrl_start):
            m.d.sync += reg_ctrl_start.eq(0)
        with m.If(reg_prog_strobe):
            m.d.sync += reg_prog_strobe.eq(0)

        # Autonomous hardware bank swap pulse from top FSM
        with m.If(self.hw_bank_swap):
            m.d.sync += [
                reg_active_bank.eq(~reg_active_bank),
                reg_bank_ready.eq(0),
            ]

        # -------------------------------------------------------------
        # Write State Machine
        # -------------------------------------------------------------
        aw_latched = Signal(1, reset=0)
        w_latched = Signal(1, reset=0)
        write_addr = Signal(self.addr_w)
        write_data = Signal(self.data_w)

        # awready & wready handshakes
        m.d.comb += [
            self.awready.eq(~aw_latched),
            self.wready.eq(~w_latched),
        ]

        with m.If(self.awvalid & ~aw_latched):
            m.d.sync += [
                aw_latched.eq(1),
                write_addr.eq(self.awaddr),
            ]
        with m.If(self.wvalid & ~w_latched):
            m.d.sync += [
                w_latched.eq(1),
                write_data.eq(self.wdata),
            ]

        with m.If(aw_latched & w_latched & ~self.bvalid):
            # Execute write
            m.d.sync += self.bvalid.eq(1)
            # Match offset bits [7:0]
            with m.Switch(write_addr[0:8]):
                with m.Case(REG_CTRL):
                    m.d.sync += [
                        reg_ctrl_start.eq(write_data[0]),
                        reg_ctrl_soft_reset.eq(write_data[1]),
                        reg_ctrl_accumulate.eq(write_data[2]),
                        reg_ctrl_ie.eq(write_data[3]),
                        reg_auto_swap_en.eq(write_data[5]),
                    ]
                    # Bit 4: arm_next_bank
                    with m.If(write_data[4]):
                        m.d.sync += reg_bank_ready.eq(1)
                    # Bit 6: manual_swap
                    with m.If(write_data[6]):
                        m.d.sync += reg_active_bank.eq(~reg_active_bank)
                with m.Case(REG_IN_FEATURES):
                    m.d.sync += reg_in_feat.eq(write_data[0:16])
                with m.Case(REG_OUT_FEATURES):
                    m.d.sync += reg_out_feat.eq(write_data[0:16])
                with m.Case(REG_TILE_MASK):
                    m.d.sync += reg_tile_mask.eq(write_data[0 : self.num_tiles])
                with m.Case(REG_PROG_ADDR):
                    m.d.sync += [
                        reg_prog_tile_idx.eq(write_data[0:tidx_w]),
                        reg_prog_mem_type.eq(write_data[8:10]),
                        reg_prog_addr.eq(write_data[16:32]),
                    ]
                with m.Case(REG_PROG_DATA):
                    m.d.sync += reg_prog_data.eq(write_data)
                with m.Case(REG_PROG_CTRL):
                    m.d.sync += reg_prog_strobe.eq(write_data[0])

        with m.Elif(self.bvalid & self.bready):
            m.d.sync += [
                self.bvalid.eq(0),
                aw_latched.eq(0),
                w_latched.eq(0),
            ]

        # -------------------------------------------------------------
        # Read State Machine
        # -------------------------------------------------------------
        ar_latched = Signal(1, reset=0)
        read_addr = Signal(self.addr_w)

        m.d.comb += self.arready.eq(~ar_latched)

        with m.If(self.arvalid & ~ar_latched):
            m.d.sync += [
                ar_latched.eq(1),
                read_addr.eq(self.araddr),
            ]

        with m.If(ar_latched & ~self.rvalid):
            m.d.sync += self.rvalid.eq(1)
            with m.Switch(read_addr[0:8]):
                with m.Case(REG_CTRL):
                    m.d.sync += self.rdata.eq(
                        Cat(
                            reg_ctrl_start,
                            reg_ctrl_soft_reset,
                            reg_ctrl_accumulate,
                            reg_ctrl_ie,
                            reg_bank_ready,
                            reg_auto_swap_en,
                            Const(0, 26),
                        )
                    )
                with m.Case(REG_STATUS):
                    m.d.sync += self.rdata.eq(
                        Cat(
                            self.status_busy,
                            self.status_done,
                            reg_active_bank,
                            reg_bank_ready,
                            Const(0, 4),
                            Const(self.num_tiles, 8),
                            Const(0, 16),
                        )
                    )
                with m.Case(REG_IN_FEATURES):
                    m.d.sync += self.rdata.eq(reg_in_feat)
                with m.Case(REG_OUT_FEATURES):
                    m.d.sync += self.rdata.eq(reg_out_feat)
                with m.Case(REG_TILE_MASK):
                    m.d.sync += self.rdata.eq(reg_tile_mask)
                with m.Case(REG_CYCLE_COUNT):
                    m.d.sync += self.rdata.eq(self.cycle_count)
                with m.Case(REG_PROG_ADDR):
                    m.d.sync += self.rdata.eq(
                        Cat(
                            reg_prog_tile_idx,
                            Const(0, 8 - tidx_w),
                            reg_prog_mem_type,
                            Const(0, 6),
                            reg_prog_addr,
                        )
                    )
                with m.Case(REG_PROG_DATA):
                    m.d.sync += self.rdata.eq(reg_prog_data)
                with m.Case(REG_PROG_CTRL):
                    m.d.sync += self.rdata.eq(reg_prog_strobe)
                with m.Default():
                    m.d.sync += self.rdata.eq(0)

        with m.Elif(self.rvalid & self.rready):
            m.d.sync += [
                self.rvalid.eq(0),
                ar_latched.eq(0),
            ]

        return m
