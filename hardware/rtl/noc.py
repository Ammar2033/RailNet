"""NoC Interconnect and Distribution Network for RailNet Multi-Tile Grid.

Provides:
- PipelinedBroadcaster: Pipelined broadcast fabric delivering input activations
  and route indices across grid tiles with minimal fan-out capacitance.
- ResultGatherConcentrator: Pipelined collection arbiter that reads Stage-B
  results from completed tiles and streams them sequentially onto AXI4-Stream.
"""

from amaranth import Cat, Const, Module, Mux, Signal, signed, unsigned
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from hardware.rtl.int8_tile import ACC_W, ACT_W


class PipelinedBroadcaster(wiring.Component):
    """Pipelined Broadcast Distribution for N Tiles.

    Registers input activations and control pulses, broadcasting to all tile
    ports without long unbuffered combinatorial paths.
    """

    def __init__(self, num_tiles: int = 4):
        self.num_tiles = num_tiles

        ports = {
            "in_x": In(signed(ACT_W)),
            "in_valid": In(1),
            "in_flush": In(1),
            "in_start_reduction": In(1),
        }
        for i in range(num_tiles):
            ports[f"out_x_{i}"] = Out(signed(ACT_W))
            ports[f"out_valid_{i}"] = Out(1)
            ports[f"out_flush_{i}"] = Out(1)
            ports[f"out_start_reduction_{i}"] = Out(1)

        super().__init__(ports)

    def elaborate(self, platform):
        m = Module()

        # Pipeline register stage for broadcast
        pipe_x = Signal(signed(ACT_W), reset=0)
        pipe_valid = Signal(1, reset=0)
        pipe_flush = Signal(1, reset=0)
        pipe_start_red = Signal(1, reset=0)

        m.d.sync += [
            pipe_x.eq(self.in_x),
            pipe_valid.eq(self.in_valid),
            pipe_flush.eq(self.in_flush),
            pipe_start_red.eq(self.in_start_reduction),
        ]

        for i in range(self.num_tiles):
            m.d.comb += [
                getattr(self, f"out_x_{i}").eq(pipe_x),
                getattr(self, f"out_valid_{i}").eq(pipe_valid),
                getattr(self, f"out_flush_{i}").eq(pipe_flush),
                getattr(self, f"out_start_reduction_{i}").eq(pipe_start_red),
            ]

        return m


class ResultGatherConcentrator(wiring.Component):
    """Gathers computed results from completed tiles and streams them onto AXI4-Stream.

    Ports:
        tile_done_{i}: Done flag from each tile.
        tile_y_{i}: 32-bit output result from each tile.
        tile_mask: Bitmask of active tiles to collect from.
        start_gather: Trigger signal to begin collection once tiles finish.
        busy: Held 1 while gathering/transmitting.
        all_gathered: Pulsed 1 when all active tile results have been transmitted.
        m_axis_tdata: AXI4-Stream data output (32-bit).
        m_axis_tvalid: AXI4-Stream valid.
        m_axis_tready: AXI4-Stream ready from downstream receiver.
        m_axis_tlast: AXI4-Stream last packet flag.
    """

    def __init__(self, num_tiles: int = 4):
        self.num_tiles = num_tiles

        ports = {
            "tile_mask": In(num_tiles),
            "start_gather": In(1),
            "busy": Out(1),
            "all_gathered": Out(1),
            "m_axis_tdata": Out(signed(ACC_W)),
            "m_axis_tvalid": Out(1),
            "m_axis_tready": In(1),
            "m_axis_tlast": Out(1),
        }
        for i in range(num_tiles):
            ports[f"tile_done_{i}"] = In(1)
            ports[f"tile_y_{i}"] = In(signed(ACC_W))

        super().__init__(ports)

    def elaborate(self, platform):
        m = Module()

        idx = Signal(range(self.num_tiles + 1), reset=0)
        gathering = Signal(1, reset=0)
        tvalid = Signal(1, reset=0)
        tlast = Signal(1, reset=0)
        tdata = Signal(signed(ACC_W), reset=0)
        done_pulse = Signal(1, reset=0)

        m.d.comb += [
            self.busy.eq(gathering),
            self.m_axis_tvalid.eq(tvalid),
            self.m_axis_tlast.eq(tlast),
            self.m_axis_tdata.eq(tdata),
            self.all_gathered.eq(done_pulse),
        ]

        m.d.sync += done_pulse.eq(0)

        # Helper to find the last active tile index
        last_active_idx = Signal(range(self.num_tiles), reset=self.num_tiles - 1)
        for i in range(self.num_tiles):
            with m.If(self.tile_mask[i]):
                m.d.comb += last_active_idx.eq(i)

        # Mux to select current tile_y based on idx
        current_y = Signal(signed(ACC_W))
        for i in range(self.num_tiles):
            with m.If(idx == i):
                m.d.comb += current_y.eq(getattr(self, f"tile_y_{i}"))

        with m.If(self.start_gather & ~gathering):
            m.d.sync += [
                gathering.eq(1),
                idx.eq(0),
                tvalid.eq(0),
                tlast.eq(0),
            ]

        with m.Elif(gathering):
            # If current output was acknowledged, or not valid yet
            with m.If(~tvalid | self.m_axis_tready):
                with m.If(idx < self.num_tiles):
                    # Check if tile idx is active in tile_mask
                    is_active = self.tile_mask.bit_select(idx, 1)
                    with m.If(is_active):
                        m.d.sync += [
                            tdata.eq(current_y),
                            tvalid.eq(1),
                            tlast.eq(idx == last_active_idx),
                            idx.eq(idx + 1),
                        ]
                    with m.Else():
                        # Skip inactive tile: clear valid & last so no phantom duplicate is emitted
                        m.d.sync += [
                            tvalid.eq(0),
                            tlast.eq(0),
                            idx.eq(idx + 1),
                        ]
                with m.Else():
                    # All tiles processed
                    m.d.sync += [
                        gathering.eq(0),
                        tvalid.eq(0),
                        tlast.eq(0),
                        done_pulse.eq(1),
                    ]

        return m
