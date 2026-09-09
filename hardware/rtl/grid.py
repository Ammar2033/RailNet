"""Multi-Tile Compute Grid for RailNet Accelerator.

Provides:
- RailNetGrid: Parametric 2D array of RailNetInt8Tile instances with local
  Weight-in-Silicon route BRAMs per tile for parallel GEMM execution.
"""

from amaranth import Cat, Const, Module, Mux, Signal, signed, unsigned
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

from hardware.rtl.int8_tile import (
    ACC_W,
    ACT_W,
    CODEBOOK_DEPTH,
    RAIL_W,
    RAILS_DEFAULT,
    RailNetInt8Tile,
)

ROUTE_DEPTH_DEFAULT = 512


class RailNetGrid(wiring.Component):
    """Parametric Multi-Tile Compute Grid.

    Instantiates N independent INT8 tiles, each equipped with its own local
    Route Table BRAM storing weight routes for one output neuron slice.
    """

    def __init__(
        self,
        num_tiles: int = 4,
        rails: int = RAILS_DEFAULT,
        codebook_depth: int = 64,
        route_depth: int = ROUTE_DEPTH_DEFAULT,
    ):
        self.num_tiles = num_tiles
        self.rails = rails
        self.codebook_depth = codebook_depth
        self.route_depth = route_depth
        self.tidx_w = max(1, (num_tiles - 1).bit_length())

        ports = {
            # Global reset / flush
            "flush": In(1),
            "start_reduction": In(1),
            "all_done": Out(1),
            # Programming Bus (targets selected tile via prog_tile_idx)
            "prog_tile_idx": In(self.tidx_w),
            "active_bank": In(1, init=0),
            "prog_bank": In(1, init=0),
            # 1) Program Route RAM
            "prog_route_en": In(1),
            "prog_route_addr": In(max(1, (route_depth - 1).bit_length())),
            "prog_route_data": In(16),
            # 2) Program Codebook
            "prog_cb_en": In(1),
            "prog_cb_addr": In(16),
            "prog_cb_data": In(27),
            # 3) Program Rails
            "prog_rail_en": In(1),
            "prog_rail_addr": In(max(1, (rails - 1).bit_length())),
            "prog_rail_data": In(signed(RAIL_W)),
        }

        # Per-tile inputs/outputs
        for i in range(num_tiles):
            ports[f"in_x_{i}"] = In(signed(ACT_W))
            ports[f"in_valid_{i}"] = In(1)
            ports[f"tile_done_{i}"] = Out(1)
            ports[f"tile_busy_{i}"] = Out(1)
            ports[f"tile_y_{i}"] = Out(signed(ACC_W))

        super().__init__(ports)

    def elaborate(self, platform):
        m = Module()

        done_signals = []
        r_addr_w = max(1, (self.route_depth - 1).bit_length())

        for i in range(self.num_tiles):
            # 1. Instantiate local Route Table BRAM for this tile (double depth for ping-pong)
            route_mem = Memory(shape=unsigned(16), depth=2 * self.route_depth, init=[0] * (2 * self.route_depth))
            setattr(m.submodules, f"route_mem_{i}", route_mem)

            route_rd = route_mem.read_port(domain="sync")
            route_wr = route_mem.write_port()

            # Input activation index counter k
            k_counter = Signal(range(self.route_depth), reset=0)
            in_valid = getattr(self, f"in_valid_{i}")
            in_x = getattr(self, f"in_x_{i}")

            # Synchronous pipeline register to align input activation with 1-cycle SRAM read latency
            x_q = Signal(signed(ACT_W), reset=0)
            valid_q = Signal(1, reset=0)

            with m.If(self.flush | self.start_reduction):
                m.d.sync += [
                    k_counter.eq(0),
                    valid_q.eq(0),
                ]
            with m.Elif(in_valid):
                with m.If(k_counter < self.route_depth - 1):
                    m.d.sync += k_counter.eq(k_counter + 1)
                with m.Else():
                    m.d.sync += k_counter.eq(0)
                m.d.sync += [
                    x_q.eq(in_x),
                    valid_q.eq(1),
                ]
            with m.Else():
                m.d.sync += valid_q.eq(0)

            m.d.comb += route_rd.addr.eq(Cat(k_counter[:r_addr_w], self.active_bank))

            # Route RAM programming
            is_targeted = self.prog_tile_idx == i
            m.d.comb += [
                route_wr.addr.eq(Cat(self.prog_route_addr[:r_addr_w], self.prog_bank)),
                route_wr.data.eq(self.prog_route_data),
                route_wr.en.eq(self.prog_route_en & is_targeted),
            ]

            # 2. Instantiate Tile
            tile = RailNetInt8Tile(rails=self.rails, codebook_depth=self.codebook_depth)
            setattr(m.submodules, f"tile_{i}", tile)

            # Connect tile inputs
            m.d.comb += [
                tile.x.eq(x_q),
                tile.route_id.eq(route_rd.data),
                tile.valid_in.eq(valid_q),
                tile.flush.eq(self.flush),
                tile.start_reduction.eq(self.start_reduction),
                tile.active_bank.eq(self.active_bank),
                tile.prog_bank.eq(self.prog_bank),
                getattr(self, f"tile_done_{i}").eq(tile.done),
                getattr(self, f"tile_busy_{i}").eq(tile.busy),
                getattr(self, f"tile_y_{i}").eq(tile.y),
                # Programming
                tile.prog_cb_en.eq(self.prog_cb_en & is_targeted),
                tile.prog_cb_addr.eq(self.prog_cb_addr),
                tile.prog_cb_data.eq(self.prog_cb_data),
                tile.prog_rail_en.eq(self.prog_rail_en & is_targeted),
                tile.prog_rail_addr.eq(self.prog_rail_addr),
                tile.prog_rail_data.eq(self.prog_rail_data),
            ]

            done_signals.append(tile.done)

        # all_done flag: 1 when all tiles assert done
        all_d = Signal(1)
        m.d.comb += all_d.eq(Cat(*done_signals).all())
        m.d.comb += self.all_done.eq(all_d)

        return m
