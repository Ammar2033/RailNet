"""Stage-A routing/gather tile vs a dense MAC inner loop — the ADR 0001 gate.

Both are streaming inner loops that fold one output feature:

  DenseInner : per cycle   acc += x * w                 (one int16xint16 multiply)
  StageA     : per term    G[rail] += (sign ? -x : x)   (no multiply; a RAILS-wide
               indexed read-modify-write — "the gather")

The question the synthesis answers: is StageA's route decode + indexed RMW +
the RAILS-deep G accumulator cheaper in fabric than one multiplier + one
accumulator? Fixed-point (int16 act, int32 acc) keeps it synthesizable; a real
BF16 datapath is a refinement, but multiplier-vs-adder area is the crux.

Amaranth 0.5.
"""

from amaranth import Array, Module, Mux, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

ACT_W = 16
ACC_W = 32


class DenseInner(wiring.Component):
    """acc += x * w per valid cycle; `last` also emits y = acc + x*w."""

    x: In(signed(ACT_W))
    w: In(signed(ACT_W))
    valid: In(1)
    last: In(1)
    y: Out(signed(ACC_W))
    y_valid: Out(1)

    def elaborate(self, platform):
        m = Module()
        acc = Signal(signed(ACC_W))
        nxt = Signal(signed(ACC_W))
        m.d.comb += nxt.eq(acc + self.x * self.w)
        m.d.sync += self.y_valid.eq(0)
        with m.If(self.valid):
            m.d.sync += acc.eq(nxt)
            with m.If(self.last):
                m.d.sync += [self.y.eq(nxt), self.y_valid.eq(1), acc.eq(0)]
        return m


class StageA(wiring.Component):
    """One term per cycle: G[rail] += (sign ? -x : x). `flush` clears G.
    Read G out via `g_addr` / `g_data`."""

    def __init__(self, rails: int = 96):
        self.rails = rails
        self.ridx_w = max(1, (rails - 1).bit_length())
        super().__init__(
            {
                "x": In(signed(ACT_W)),
                "rail": In(self.ridx_w),
                "sign": In(1),
                "valid": In(1),
                "flush": In(1),
                "g_addr": In(self.ridx_w),
                "g_data": Out(signed(ACC_W)),
            }
        )

    def elaborate(self, platform):
        m = Module()
        g = Array(Signal(signed(ACC_W), name=f"g{r}") for r in range(self.rails))
        delta = Signal(signed(ACT_W + 1))
        m.d.comb += delta.eq(Mux(self.sign, -self.x, self.x))

        with m.If(self.flush):
            for r in range(self.rails):
                m.d.sync += g[r].eq(0)
        with m.Elif(self.valid):
            with m.Switch(self.rail):
                for r in range(self.rails):
                    with m.Case(r):
                        m.d.sync += g[r].eq(g[r] + delta)

        m.d.comb += self.g_data.eq(g[self.g_addr])
        return m


class StageABram(wiring.Component):
    """Same as StageA but G lives in a 1R1W memory (a BRAM), so no 96 FF bank
    and no 96-way switch — read G[rail], add ±x, write back. Pipelined RMW:
    the caller must not issue two terms for the same rail back-to-back (RailNet
    routes use distinct rails per weight, so within a weight this holds; a real
    controller inserts a bubble between weights)."""

    def __init__(self, rails: int = 96):
        self.rails = rails
        self.ridx_w = max(1, (rails - 1).bit_length())
        super().__init__(
            {
                "x": In(signed(ACT_W)),
                "rail": In(self.ridx_w),
                "sign": In(1),
                "valid": In(1),
                "flush": In(1),
                "g_addr": In(self.ridx_w),
                "g_data": Out(signed(ACC_W)),
            }
        )

    def elaborate(self, platform):
        m = Module()
        m.submodules.mem = mem = Memory(
            shape=signed(ACC_W), depth=self.rails, init=[0] * self.rails
        )
        rd = mem.read_port(domain="sync")
        wr = mem.write_port()
        rd2 = mem.read_port(domain="comb")  # async read for streaming G out
        m.d.comb += [rd2.addr.eq(self.g_addr), self.g_data.eq(rd2.data)]

        delta = Signal(signed(ACT_W + 1))
        m.d.comb += delta.eq(Mux(self.sign, -self.x, self.x))

        # multi-cycle flush: sweep the memory writing zeros
        fcnt = Signal(range(self.rails + 1))
        flushing = Signal()
        with m.If(self.flush):
            m.d.sync += [flushing.eq(1), fcnt.eq(0)]
        with m.Elif(flushing):
            m.d.sync += fcnt.eq(fcnt + 1)
            with m.If(fcnt == self.rails - 1):
                m.d.sync += flushing.eq(0)

        # RMW pipeline: cycle N read G[rail]; cycle N+1 write G[rail]+delta
        r1 = Signal(self.ridx_w)
        d1 = Signal(signed(ACT_W + 1))
        v1 = Signal()
        m.d.comb += rd.addr.eq(self.rail)
        m.d.sync += [r1.eq(self.rail), d1.eq(delta), v1.eq(self.valid & ~flushing & ~self.flush)]

        with m.If(flushing):
            m.d.comb += [wr.addr.eq(fcnt), wr.data.eq(0), wr.en.eq(1)]
        with m.Else():
            m.d.comb += [wr.addr.eq(r1), wr.data.eq(rd.data + d1), wr.en.eq(v1)]
        return m


class StageB(wiring.Component):
    """Y = sum_r rails[r] * G[r] — one shared multiplier, RAILS cycles."""

    rail_val: In(signed(ACT_W))
    g: In(signed(ACC_W))
    valid: In(1)
    last: In(1)
    y: Out(signed(ACC_W + ACT_W))
    y_valid: Out(1)

    def elaborate(self, platform):
        m = Module()
        acc = Signal(signed(ACC_W + ACT_W))
        nxt = Signal(signed(ACC_W + ACT_W))
        m.d.comb += nxt.eq(acc + self.rail_val * self.g)
        m.d.sync += self.y_valid.eq(0)
        with m.If(self.valid):
            m.d.sync += acc.eq(nxt)
            with m.If(self.last):
                m.d.sync += [self.y.eq(nxt), self.y_valid.eq(1), acc.eq(0)]
        return m


__all__ = ["ACC_W", "ACT_W", "DenseInner", "StageA", "StageABram", "StageB"]
