"""BF16/FP32 RailNet Accelerator Tile (The Taalas-vs-RailNet Thesis).

Implements the 3-lane throughput-matched Stage-A gather + Stage-B reduction
architecture in pure Amaranth 0.5:
- RouteDecoder: decodes 16-bit route_ids into up to 3 parallel (rail, sign, valid) terms.
- StageABf16Bram: 96-deep 32-bit BRAM accumulator with CombFp32Adder
  and single-cycle forwarding/bypass network (zero bubbles on same-rail hazard).
- StageBBf16: reads 3 Stage-A lanes, computes (G0[r]+G1[r]+G2[r]) * R[r], accumulates,
  and emits a bit-accurate BF16 output Y[j].
- RailNetFullTile: Top-level integrated tile with control FSM.

Amaranth 0.5.
"""

from amaranth import Cat, Const, Module, Mux, Signal, unsigned
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

from hardware.rtl.fp32 import Bf16ToFp32, CombFp32Adder, Fp32Multiplier, Fp32ToBf16

RAILS_DEFAULT = 96
CODEBOOK_DEPTH = 2048


class RouteDecoder(wiring.Component):
    """Decodes a 16-bit route ID into up to 3 parallel terms for 3 Stage-A lanes.

    Codebook entry format (27 bits):
      [0:7]   term 0 rail (0..95)
      [7]     term 0 sign (0: +, 1: -)
      [8]     term 0 active (1: active)
      [9:16]  term 1 rail
      [16]    term 1 sign
      [17]    term 1 active
      [18:25] term 2 rail
      [25]    term 2 sign
      [26]    term 2 active
    """

    def __init__(self, rails: int = RAILS_DEFAULT, depth: int = CODEBOOK_DEPTH):
        self.rails = rails
        self.depth = depth
        self.ridx_w = max(1, (rails - 1).bit_length())

        super().__init__(
            {
                # Stream input
                "route_id": In(16),
                "valid_in": In(1),
                # Programming interface (for loading tensor routing table)
                "prog_en": In(1),
                "prog_addr": In(16),
                "prog_data": In(27),
                # Decoded terms for 3 Stage-A lanes
                "t0_rail": Out(self.ridx_w),
                "t0_sign": Out(1),
                "t0_valid": Out(1),
                "t1_rail": Out(self.ridx_w),
                "t1_sign": Out(1),
                "t1_valid": Out(1),
                "t2_rail": Out(self.ridx_w),
                "t2_sign": Out(1),
                "t2_valid": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        mem = Memory(shape=unsigned(27), depth=self.depth, init=[0] * self.depth)
        m.submodules.codebook = mem

        rd = mem.read_port(domain="comb")
        wr = mem.write_port()

        # Programming port
        m.d.comb += [
            wr.addr.eq(self.prog_addr),
            wr.data.eq(self.prog_data),
            wr.en.eq(self.prog_en),
        ]

        # Decode port
        m.d.comb += rd.addr.eq(self.route_id[0:max(1, (self.depth - 1).bit_length())])

        entry = rd.data
        m.d.comb += [
            self.t0_rail.eq(entry[0:self.ridx_w]),
            self.t0_sign.eq(entry[7]),
            self.t0_valid.eq(entry[8] & self.valid_in),
            self.t1_rail.eq(entry[9:9 + self.ridx_w]),
            self.t1_sign.eq(entry[16]),
            self.t1_valid.eq(entry[17] & self.valid_in),
            self.t2_rail.eq(entry[18:18 + self.ridx_w]),
            self.t2_sign.eq(entry[25]),
            self.t2_valid.eq(entry[26] & self.valid_in),
        ]

        return m


class StageABf16Bram(wiring.Component):
    """One Stage-A lane: accumulates BF16 inputs into private 96x32b FP32 G memory.

    Features:
    - CombFp32Adder for direct, zero-stall accumulation.
    - Forwarding/bypass register for consecutive accesses to the same rail.
    - Multi-cycle flush to clear memory.
    - Asynchronous readout port for Stage-B.
    """

    def __init__(self, rails: int = RAILS_DEFAULT):
        self.rails = rails
        self.ridx_w = max(1, (rails - 1).bit_length())

        super().__init__(
            {
                # Term input
                "x_bf16": In(16),
                "rail": In(self.ridx_w),
                "sign": In(1),
                "valid": In(1),
                "flush": In(1),
                "flushing": Out(1),
                # Stage-B readout port
                "g_addr": In(self.ridx_w),
                "g_data": Out(unsigned(32)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        mem = Memory(shape=unsigned(32), depth=self.rails, init=[0] * self.rails)
        m.submodules.mem = mem

        rd_sync = mem.read_port(domain="sync")
        wr = mem.write_port()
        rd_async = mem.read_port(domain="comb")

        m.d.comb += [
            rd_async.addr.eq(self.g_addr),
            self.g_data.eq(rd_async.data),
        ]

        # Multi-cycle flush logic
        fcnt = Signal(range(self.rails + 1))
        flushing_sig = Signal()
        m.d.comb += self.flushing.eq(flushing_sig)

        with m.If(self.flush):
            m.d.sync += [flushing_sig.eq(1), fcnt.eq(0)]
        with m.Elif(flushing_sig):
            m.d.sync += fcnt.eq(fcnt + 1)
            with m.If(fcnt == self.rails - 1):
                m.d.sync += flushing_sig.eq(0)

        # BF16 to FP32 conversion on incoming x
        bf16_conv = Bf16ToFp32()
        m.submodules.bf16_conv = bf16_conv
        m.d.comb += bf16_conv.bf16.eq(self.x_bf16)

        # Single-cycle FP32 adder
        adder = CombFp32Adder()
        m.submodules.adder = adder

        # Pipeline register: Cycle N sends read, Cycle N+1 performs add & write
        r1_rail = Signal(self.ridx_w)
        r1_sign = Signal()
        r1_x_fp32 = Signal(32)
        r1_valid = Signal()

        m.d.comb += rd_sync.addr.eq(self.rail)
        m.d.sync += [
            r1_rail.eq(self.rail),
            r1_sign.eq(self.sign),
            r1_x_fp32.eq(bf16_conv.fp32),
            r1_valid.eq(self.valid & ~flushing_sig & ~self.flush),
        ]

        # Forwarding writeback register
        wb_valid = Signal()
        wb_rail = Signal(self.ridx_w)
        wb_data = Signal(32)

        m.d.sync += [
            wb_valid.eq(r1_valid & ~flushing_sig),
            wb_rail.eq(r1_rail),
            wb_data.eq(adder.res),
        ]

        # FORWARDING NETWORK:
        # If r1_rail matches the rail updated in the previous cycle, bypass BRAM!
        eff_g_val = Signal(32)
        with m.If(wb_valid & (wb_rail == r1_rail)):
            m.d.comb += eff_g_val.eq(wb_data)
        with m.Else():
            m.d.comb += eff_g_val.eq(rd_sync.data)

        # Feed adder
        m.d.comb += [
            adder.a.eq(eff_g_val),
            adder.b.eq(r1_x_fp32),
            adder.sub.eq(r1_sign),
        ]

        # Memory write port
        with m.If(flushing_sig):
            m.d.comb += [wr.addr.eq(fcnt), wr.data.eq(0), wr.en.eq(1)]
        with m.Else():
            m.d.comb += [
                wr.addr.eq(r1_rail),
                wr.data.eq(adder.res),
                wr.en.eq(r1_valid),
            ]

        return m


class StageBBf16(wiring.Component):
    """Stage-B reduction: reads G0, G1, G2 across all rails, computes:
       Y = sum_r (G0[r] + G1[r] + G2[r]) * Rail[r]
    and emits a BF16 output Y[j].
    """

    def __init__(self, rails: int = RAILS_DEFAULT):
        self.rails = rails
        self.ridx_w = max(1, (rails - 1).bit_length())

        super().__init__(
            {
                # Inputs from the 3 Stage-A G memories
                "g0": In(unsigned(32)),
                "g1": In(unsigned(32)),
                "g2": In(unsigned(32)),
                "rail_val_bf16": In(16),
                "g_addr": Out(self.ridx_w),
                # Control
                "start": In(1),
                "busy": Out(1),
                # Output
                "y_bf16": Out(16),
                "y_valid": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        add01 = CombFp32Adder()
        add012 = CombFp32Adder()
        rail_conv = Bf16ToFp32()
        mult = Fp32Multiplier()
        acc_adder = CombFp32Adder()
        rnd_to_bf16 = Fp32ToBf16()

        m.submodules.add01 = add01
        m.submodules.add012 = add012
        m.submodules.rail_conv = rail_conv
        m.submodules.mult = mult
        m.submodules.acc_adder = acc_adder
        m.submodules.rnd_to_bf16 = rnd_to_bf16

        m.d.comb += rail_conv.bf16.eq(self.rail_val_bf16)

        # FSM / Sequencer
        rail_cnt = Signal(range(self.rails + 1))
        running = Signal()
        m.d.comb += [
            self.g_addr.eq(rail_cnt[0:self.ridx_w]),
            self.busy.eq(running),
        ]

        with m.If(self.start):
            m.d.sync += [running.eq(1), rail_cnt.eq(0)]
        with m.Elif(running):
            m.d.sync += rail_cnt.eq(rail_cnt + 1)
            with m.If(rail_cnt == self.rails - 1):
                m.d.sync += running.eq(0)

        # Add g0 + g1
        m.d.comb += [
            add01.a.eq(self.g0),
            add01.b.eq(self.g1),
            add01.sub.eq(0),
        ]

        # Add (g0 + g1) + g2
        m.d.comb += [
            add012.a.eq(add01.res),
            add012.b.eq(self.g2),
            add012.sub.eq(0),
        ]

        # Feed Multiplier: G_sum * rail_fp32
        m.d.comb += [
            mult.a.eq(add012.res),
            mult.b.eq(rail_conv.fp32),
            mult.valid_in.eq(running),
        ]

        # Multiplier has 2 pipeline stages. Track rail index through multiplier pipeline:
        r_tag_0 = Signal(self.ridx_w)
        r_tag_1 = Signal(self.ridx_w)
        m.d.sync += [
            r_tag_0.eq(rail_cnt[0:self.ridx_w]),
            r_tag_1.eq(r_tag_0),
        ]

        # Accumulator: acc_val += mult.res
        acc_val = Signal(32)
        self.acc_val = acc_val
        self.mult = mult
        self.acc_adder = acc_adder
        m.d.comb += [
            acc_adder.a.eq(acc_val),
            acc_adder.b.eq(mult.res),
            acc_adder.sub.eq(0),
        ]

        m.d.sync += self.y_valid.eq(0)

        with m.If(self.start):
            m.d.sync += acc_val.eq(0)
        with m.Elif(mult.valid_out):
            m.d.sync += acc_val.eq(acc_adder.res)
            # When the last rail is accumulated, emit output
            with m.If(r_tag_1 == self.rails - 1):
                m.d.sync += self.y_valid.eq(1)

        m.d.comb += [
            rnd_to_bf16.fp32.eq(acc_val),
            self.y_bf16.eq(rnd_to_bf16.bf16),
        ]

        return m


class RailNetFullTile(wiring.Component):
    """Full RailNet BF16 Accelerator Tile:
    - RouteDecoder
    - 3x StageABf16Bram
    - StageBBf16
    """

    def __init__(self, rails: int = RAILS_DEFAULT, codebook_depth: int = CODEBOOK_DEPTH):
        self.rails = rails
        self.codebook_depth = codebook_depth
        self.ridx_w = max(1, (rails - 1).bit_length())

        super().__init__(
            {
                # Programming
                "prog_en": In(1),
                "prog_addr": In(16),
                "prog_data": In(27),
                # Execution
                "flush": In(1),
                "x_bf16": In(16),
                "route_id": In(16),
                "stream_valid": In(1),
                # Stage-B rail stream
                "reduce_start": In(1),
                "rail_val_bf16": In(16),
                "rail_addr": Out(self.ridx_w),
                "stage_b_busy": Out(1),
                # Result
                "y_bf16": Out(16),
                "y_valid": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        dec = RouteDecoder(rails=self.rails, depth=self.codebook_depth)
        lane0 = StageABf16Bram(rails=self.rails)
        lane1 = StageABf16Bram(rails=self.rails)
        lane2 = StageABf16Bram(rails=self.rails)
        stage_b = StageBBf16(rails=self.rails)

        self.dec = dec
        self.lane0 = lane0
        self.lane1 = lane1
        self.lane2 = lane2
        self.stage_b = stage_b

        m.submodules.dec = dec
        m.submodules.lane0 = lane0
        m.submodules.lane1 = lane1
        m.submodules.lane2 = lane2
        m.submodules.stage_b = stage_b

        # RouteDecoder connections
        m.d.comb += [
            dec.prog_en.eq(self.prog_en),
            dec.prog_addr.eq(self.prog_addr),
            dec.prog_data.eq(self.prog_data),
            dec.route_id.eq(self.route_id),
            dec.valid_in.eq(self.stream_valid),
        ]

        # Lane 0
        m.d.comb += [
            lane0.flush.eq(self.flush),
            lane0.x_bf16.eq(self.x_bf16),
            lane0.rail.eq(dec.t0_rail),
            lane0.sign.eq(dec.t0_sign),
            lane0.valid.eq(dec.t0_valid),
            lane0.g_addr.eq(stage_b.g_addr),
        ]

        # Lane 1
        m.d.comb += [
            lane1.flush.eq(self.flush),
            lane1.x_bf16.eq(self.x_bf16),
            lane1.rail.eq(dec.t1_rail),
            lane1.sign.eq(dec.t1_sign),
            lane1.valid.eq(dec.t1_valid),
            lane1.g_addr.eq(stage_b.g_addr),
        ]

        # Lane 2
        m.d.comb += [
            lane2.flush.eq(self.flush),
            lane2.x_bf16.eq(self.x_bf16),
            lane2.rail.eq(dec.t2_rail),
            lane2.sign.eq(dec.t2_sign),
            lane2.valid.eq(dec.t2_valid),
            lane2.g_addr.eq(stage_b.g_addr),
        ]

        # Stage B
        m.d.comb += [
            stage_b.g0.eq(lane0.g_data),
            stage_b.g1.eq(lane1.g_data),
            stage_b.g2.eq(lane2.g_data),
            stage_b.rail_val_bf16.eq(self.rail_val_bf16),
            stage_b.start.eq(self.reduce_start),
            self.rail_addr.eq(stage_b.g_addr),
            self.stage_b_busy.eq(stage_b.busy),
            self.y_bf16.eq(stage_b.y_bf16),
            self.y_valid.eq(stage_b.y_valid),
        ]

        return m


__all__ = ["RailNetFullTile", "RouteDecoder", "StageABf16Bram", "StageBBf16"]
