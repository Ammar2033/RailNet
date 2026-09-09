"""Top-Level Integrated Accelerator Architecture (RailNetTop).

Combines:
- AxiLiteCsr: Memory-mapped Control & Status Registers (CSR)
- AxiStream: Input activation DMA streaming (s_axis) and output result streaming (m_axis)
- PipelinedBroadcaster: Fan-out distribution across tiles
- RailNetGrid: Parametric 2D Multi-Tile INT8 Compute Array
- ResultGatherConcentrator: Output gathering onto AXI4-Stream master
- Hardware Performance Cycle Counter
"""

from amaranth import Cat, Const, Module, Mux, Signal, signed, unsigned
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from hardware.rtl.axi import AxiLiteCsr
from hardware.rtl.grid import RailNetGrid
from hardware.rtl.int8_tile import ACC_W, ACT_W, RAILS_DEFAULT
from hardware.rtl.noc import PipelinedBroadcaster, ResultGatherConcentrator


class RailNetTop(wiring.Component):
    """Top-Level RailNet Hardware Accelerator.

    Standard SoC / FPGA / ASIC Top-level wrapping an N-tile compute grid with
    standard AXI4-Lite (CSR) and AXI4-Stream (DMA) ports.
    """

    def __init__(
        self,
        num_tiles: int = 4,
        rails: int = RAILS_DEFAULT,
        codebook_depth: int = 64,
        route_depth: int = 512,
    ):
        self.num_tiles = num_tiles
        self.rails = rails
        self.codebook_depth = codebook_depth
        self.route_depth = route_depth
        self.tidx_w = max(1, (num_tiles - 1).bit_length())

        super().__init__(
            {
                # -----------------------------------------------------
                # AXI4-Lite Slave Interface (CSR)
                # -----------------------------------------------------
                "s_axi_awaddr": In(12),
                "s_axi_awvalid": In(1),
                "s_axi_awready": Out(1),
                "s_axi_wdata": In(32),
                "s_axi_wstrb": In(4),
                "s_axi_wvalid": In(1),
                "s_axi_wready": Out(1),
                "s_axi_bresp": Out(2),
                "s_axi_bvalid": Out(1),
                "s_axi_bready": In(1),
                "s_axi_araddr": In(12),
                "s_axi_arvalid": In(1),
                "s_axi_arready": Out(1),
                "s_axi_rdata": Out(32),
                "s_axi_rresp": Out(2),
                "s_axi_rvalid": Out(1),
                "s_axi_rready": In(1),
                "irq": Out(1),
                # -----------------------------------------------------
                # AXI4-Stream Slave Interface (Input Activations)
                # -----------------------------------------------------
                "s_axis_tdata": In(32),
                "s_axis_tvalid": In(1),
                "s_axis_tready": Out(1),
                "s_axis_tlast": In(1),
                # -----------------------------------------------------
                # AXI4-Stream Master Interface (Output Results)
                # -----------------------------------------------------
                "m_axis_tdata": Out(signed(ACC_W)),
                "m_axis_tvalid": Out(1),
                "m_axis_tready": In(1),
                "m_axis_tlast": Out(1),
                # -----------------------------------------------------
                # Direct Weight/Route Programming Interface
                # -----------------------------------------------------
                "prog_tile_idx": In(self.tidx_w),
                "prog_route_en": In(1),
                "prog_route_addr": In(max(1, (route_depth - 1).bit_length())),
                "prog_route_data": In(16),
                "prog_cb_en": In(1),
                "prog_cb_addr": In(16),
                "prog_cb_data": In(27),
                "prog_rail_en": In(1),
                "prog_rail_addr": In(max(1, (rails - 1).bit_length())),
                "prog_rail_data": In(signed(8)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        # 1. Instantiate Submodules
        csr = AxiLiteCsr(num_tiles=self.num_tiles)
        m.submodules.csr = csr

        grid = RailNetGrid(
            num_tiles=self.num_tiles,
            rails=self.rails,
            codebook_depth=self.codebook_depth,
            route_depth=self.route_depth,
        )
        m.submodules.grid = grid

        broadcaster = PipelinedBroadcaster(num_tiles=self.num_tiles)
        m.submodules.broadcaster = broadcaster

        gather = ResultGatherConcentrator(num_tiles=self.num_tiles)
        m.submodules.gather = gather

        # 2. Connect AXI-Lite Ports to CSR
        m.d.comb += [
            csr.awaddr.eq(self.s_axi_awaddr),
            csr.awvalid.eq(self.s_axi_awvalid),
            self.s_axi_awready.eq(csr.awready),
            csr.wdata.eq(self.s_axi_wdata),
            csr.wstrb.eq(self.s_axi_wstrb),
            csr.wvalid.eq(self.s_axi_wvalid),
            self.s_axi_wready.eq(csr.wready),
            self.s_axi_bresp.eq(csr.bresp),
            self.s_axi_bvalid.eq(csr.bvalid),
            csr.bready.eq(self.s_axi_bready),
            csr.araddr.eq(self.s_axi_araddr),
            csr.arvalid.eq(self.s_axi_arvalid),
            self.s_axi_arready.eq(csr.arready),
            self.s_axi_rdata.eq(csr.rdata),
            self.s_axi_rresp.eq(csr.rresp),
            self.s_axi_rvalid.eq(csr.rvalid),
            csr.rready.eq(self.s_axi_rready),
            self.irq.eq(csr.irq),
        ]

        # 3. Connect Programming Ports to Grid (OR direct external pins with CSR in-band strobe)
        in_band_prog = csr.prog_strobe
        is_route = (csr.prog_mem_type == 0)
        is_cb = (csr.prog_mem_type == 1)
        is_rail = (csr.prog_mem_type == 2)

        route_addr_w = max(1, (self.route_depth - 1).bit_length())
        rail_addr_w = max(1, (self.rails - 1).bit_length())

        m.d.comb += [
            grid.prog_tile_idx.eq(Mux(in_band_prog, csr.prog_tile_idx, self.prog_tile_idx)),
            grid.prog_route_en.eq(self.prog_route_en | (in_band_prog & is_route)),
            grid.prog_route_addr.eq(Mux(in_band_prog & is_route, csr.prog_addr[0:route_addr_w], self.prog_route_addr)),
            grid.prog_route_data.eq(Mux(in_band_prog & is_route, csr.prog_data[0:16], self.prog_route_data)),
            grid.prog_cb_en.eq(self.prog_cb_en | (in_band_prog & is_cb)),
            grid.prog_cb_addr.eq(Mux(in_band_prog & is_cb, csr.prog_addr, self.prog_cb_addr)),
            grid.prog_cb_data.eq(Mux(in_band_prog & is_cb, csr.prog_data[0:27], self.prog_cb_data)),
            grid.prog_rail_en.eq(self.prog_rail_en | (in_band_prog & is_rail)),
            grid.prog_rail_addr.eq(Mux(in_band_prog & is_rail, csr.prog_addr[0:rail_addr_w], self.prog_rail_addr)),
            grid.prog_rail_data.eq(Mux(in_band_prog & is_rail, csr.prog_data[0:8].as_signed(), self.prog_rail_data)),
            grid.active_bank.eq(csr.active_bank),
            grid.prog_bank.eq(csr.prog_bank),
        ]

        # 4. Connect Broadcaster outputs to Grid inputs
        for i in range(self.num_tiles):
            m.d.comb += [
                getattr(grid, f"in_x_{i}").eq(getattr(broadcaster, f"out_x_{i}")),
                getattr(grid, f"in_valid_{i}").eq(getattr(broadcaster, f"out_valid_{i}")),
            ]

        # 5. Connect Grid outputs to Gather inputs
        for i in range(self.num_tiles):
            m.d.comb += [
                getattr(gather, f"tile_done_{i}").eq(getattr(grid, f"tile_done_{i}")),
                getattr(gather, f"tile_y_{i}").eq(getattr(grid, f"tile_y_{i}")),
            ]

        # Connect Gather output to m_axis
        m.d.comb += [
            self.m_axis_tdata.eq(gather.m_axis_tdata),
            self.m_axis_tvalid.eq(gather.m_axis_tvalid),
            self.m_axis_tlast.eq(gather.m_axis_tlast),
            gather.m_axis_tready.eq(self.m_axis_tready),
            gather.tile_mask.eq(csr.tile_mask),
        ]

        # 6. Top-Level Control FSM & Performance Counter
        state = Signal(range(8), reset=0)
        STATE_IDLE = 0
        STATE_FLUSH = 1
        STATE_STREAM_IN = 2
        STATE_DRAIN = 3
        STATE_START_RED = 4
        STATE_WAIT_RED = 5
        STATE_STREAM_OUT = 6
        STATE_DONE = 7

        cycle_counter = Signal(32, reset=0)
        flush_counter = Signal(range(self.rails + 10), reset=0)
        drain_counter = Signal(range(4), reset=0)
        input_counter = Signal(16, reset=0)
        busy_sig = Signal(1, reset=0)
        done_sig = Signal(1, reset=0)

        m.d.comb += [
            csr.status_busy.eq(busy_sig),
            csr.status_done.eq(done_sig),
            csr.cycle_count.eq(cycle_counter),
        ]

        # Default control signals
        m.d.comb += [
            grid.flush.eq(0),
            grid.start_reduction.eq(0),
            broadcaster.in_valid.eq(0),
            broadcaster.in_flush.eq(0),
            broadcaster.in_start_reduction.eq(0),
            broadcaster.in_x.eq(self.s_axis_tdata[0:ACT_W].as_signed()),
            gather.start_gather.eq(0),
            self.s_axis_tready.eq(0),
            csr.hw_bank_swap.eq(0),
        ]

        # Global soft-reset abort across all states
        with m.If(csr.ctrl_soft_reset):
            m.d.sync += [
                state.eq(STATE_IDLE),
                busy_sig.eq(0),
                done_sig.eq(0),
                cycle_counter.eq(0),
                flush_counter.eq(0),
                drain_counter.eq(0),
                input_counter.eq(0),
            ]
        with m.Else():
            with m.Switch(state):
                with m.Case(STATE_IDLE):
                    m.d.sync += [
                        busy_sig.eq(0),
                        done_sig.eq(0),
                    ]
                    with m.If(csr.ctrl_start):
                        with m.If(csr.ctrl_accumulate):
                            # Multi-pass chunk: keep Stage-A accumulator contents, do not flush!
                            m.d.sync += [
                                state.eq(STATE_STREAM_IN),
                                busy_sig.eq(1),
                                cycle_counter.eq(0),
                                input_counter.eq(0),
                            ]
                        with m.Else():
                            m.d.sync += [
                                state.eq(STATE_FLUSH),
                                busy_sig.eq(1),
                                cycle_counter.eq(0),
                                flush_counter.eq(0),
                                input_counter.eq(0),
                            ]

                with m.Case(STATE_FLUSH):
                    m.d.sync += cycle_counter.eq(cycle_counter + 1)
                    with m.If(flush_counter == 0):
                        m.d.comb += [
                            grid.flush.eq(1),
                            broadcaster.in_flush.eq(1),
                        ]
                        m.d.sync += flush_counter.eq(1)
                    with m.Elif(flush_counter < self.rails + 3):
                        m.d.sync += flush_counter.eq(flush_counter + 1)
                    with m.Else():
                        m.d.sync += [
                            state.eq(STATE_STREAM_IN),
                            input_counter.eq(0),
                        ]

                with m.Case(STATE_STREAM_IN):
                    m.d.sync += cycle_counter.eq(cycle_counter + 1)
                    m.d.comb += self.s_axis_tready.eq(1)

                    with m.If(self.s_axis_tvalid):
                        # Broadcast activation token element to all tiles
                        m.d.comb += [
                            broadcaster.in_valid.eq(1),
                            broadcaster.in_x.eq(self.s_axis_tdata[0:ACT_W].as_signed()),
                        ]
                        m.d.sync += input_counter.eq(input_counter + 1)

                        # Finished input stream on tlast or reached in_features
                        with m.If(self.s_axis_tlast | (input_counter + 1 >= csr.in_features)):
                            m.d.sync += [
                                state.eq(STATE_DRAIN),
                                drain_counter.eq(0),
                            ]

                with m.Case(STATE_DRAIN):
                    m.d.sync += cycle_counter.eq(cycle_counter + 1)
                    # Allow broadcaster pipeline and Stage-A BRAM write to settle
                    with m.If(drain_counter < 2):
                        m.d.sync += drain_counter.eq(drain_counter + 1)
                    with m.Else():
                        m.d.sync += state.eq(STATE_START_RED)

                with m.Case(STATE_START_RED):
                    m.d.sync += cycle_counter.eq(cycle_counter + 1)
                    m.d.comb += [
                        grid.start_reduction.eq(1),
                        broadcaster.in_start_reduction.eq(1),
                    ]
                    m.d.sync += state.eq(STATE_WAIT_RED)

                with m.Case(STATE_WAIT_RED):
                    m.d.sync += cycle_counter.eq(cycle_counter + 1)
                    with m.If(grid.all_done):
                        m.d.comb += gather.start_gather.eq(1)
                        m.d.sync += state.eq(STATE_STREAM_OUT)

                with m.Case(STATE_STREAM_OUT):
                    m.d.sync += cycle_counter.eq(cycle_counter + 1)
                    with m.If(gather.all_gathered):
                        m.d.sync += [
                            state.eq(STATE_DONE),
                            done_sig.eq(1),
                            busy_sig.eq(0),
                        ]

                with m.Case(STATE_DONE):
                    # Autonomous zero-bubble bank swap if armed
                    with m.If(csr.auto_swap_en & csr.bank_ready):
                        m.d.comb += csr.hw_bank_swap.eq(1)
                        with m.If(csr.ctrl_accumulate):
                            m.d.sync += [
                                state.eq(STATE_STREAM_IN),
                                busy_sig.eq(1),
                                done_sig.eq(0),
                                cycle_counter.eq(0),
                                input_counter.eq(0),
                            ]
                        with m.Else():
                            m.d.sync += [
                                state.eq(STATE_FLUSH),
                                busy_sig.eq(1),
                                done_sig.eq(0),
                                cycle_counter.eq(0),
                                flush_counter.eq(0),
                                input_counter.eq(0),
                            ]
                    # Remain done until next start
                    with m.Elif(csr.ctrl_start):
                        with m.If(csr.ctrl_accumulate):
                            m.d.sync += [
                                state.eq(STATE_STREAM_IN),
                                busy_sig.eq(1),
                                done_sig.eq(0),
                                cycle_counter.eq(0),
                                input_counter.eq(0),
                            ]
                        with m.Else():
                            m.d.sync += [
                                state.eq(STATE_FLUSH),
                                busy_sig.eq(1),
                                done_sig.eq(0),
                                cycle_counter.eq(0),
                                flush_counter.eq(0),
                                input_counter.eq(0),
                            ]

        return m
