"""Functional check: the RTL tiles vs a numpy golden model.

Amaranth's built-in simulator; run with pytest or directly. Proves the
synthesis numbers describe a *correct* stage-A gather, not garbage.
"""

import numpy as np
from amaranth.sim import Simulator

from hardware.rtl.tiles import DenseInner, StageA, StageABram

RAILS = 8
MAX_TERMS = 4


def _routes(rng, out_f, in_f):
    """Per weight: a set of distinct (rail, sign), 1..MAX_TERMS of them."""
    r = np.empty((out_f, in_f), dtype=object)
    for j in range(out_f):
        for i in range(in_f):
            k = rng.integers(1, MAX_TERMS + 1)
            rails = rng.choice(RAILS, size=k, replace=False)
            signs = rng.integers(0, 2, size=k)
            r[j, i] = list(zip(rails.tolist(), signs.tolist()))
    return r


def _golden_stage_a(x, routes_row):
    g = np.zeros(RAILS, dtype=np.int64)
    for i, terms in enumerate(routes_row):
        for rail, sign in terms:
            g[rail] += -int(x[i]) if sign else int(x[i])
    return g


def _run_stagea(dut_cls, x, routes_row):
    dut = dut_cls(RAILS)
    got = np.zeros(RAILS, dtype=np.int64)

    async def tb(ctx):
        ctx.set(dut.flush, 1)
        await ctx.tick()
        ctx.set(dut.flush, 0)
        for _ in range(RAILS + 3):  # let the multi-cycle flush finish
            await ctx.tick()
        for i, terms in enumerate(routes_row):
            for rail, sign in terms:
                ctx.set(dut.x, int(x[i]))
                ctx.set(dut.rail, int(rail))
                ctx.set(dut.sign, int(sign))
                ctx.set(dut.valid, 1)
                await ctx.tick()
            ctx.set(dut.valid, 0)
            await ctx.tick()  # bubble between weights (BRAM RMW hazard)
        for _ in range(3):
            await ctx.tick()
        for r in range(RAILS):
            ctx.set(dut.g_addr, r)
            await ctx.delay(1e-9)
            got[r] = ctx.get(dut.g_data)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    return got


def test_stage_a_reg_matches_golden():
    rng = np.random.default_rng(0)
    x = rng.integers(-100, 100, size=12)
    routes = _routes(rng, 1, 12)[0]
    assert np.array_equal(_run_stagea(StageA, x, routes), _golden_stage_a(x, routes))


def test_stage_a_bram_matches_golden():
    rng = np.random.default_rng(1)
    x = rng.integers(-100, 100, size=12)
    routes = _routes(rng, 1, 12)[0]
    assert np.array_equal(_run_stagea(StageABram, x, routes), _golden_stage_a(x, routes))


def test_stage_a_bram_matches_reg_on_many():
    rng = np.random.default_rng(2)
    for _ in range(8):
        x = rng.integers(-200, 200, size=rng.integers(6, 16))
        routes = _routes(rng, 1, len(x))[0]
        gold = _golden_stage_a(x, routes)
        assert np.array_equal(_run_stagea(StageA, x, routes), gold)
        assert np.array_equal(_run_stagea(StageABram, x, routes), gold)


def test_dense_inner_matches_dot():
    rng = np.random.default_rng(3)
    x = rng.integers(-100, 100, size=10)
    w = rng.integers(-100, 100, size=10)
    dut = DenseInner()
    out = {}

    async def tb(ctx):
        for i in range(10):
            ctx.set(dut.x, int(x[i]))
            ctx.set(dut.w, int(w[i]))
            ctx.set(dut.valid, 1)
            ctx.set(dut.last, int(i == 9))
            await ctx.tick()
        ctx.set(dut.valid, 0)
        await ctx.tick()
        out["y"] = ctx.get(dut.y)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    assert out["y"] == int(x @ w)


if __name__ == "__main__":
    for fn in [
        test_stage_a_reg_matches_golden,
        test_stage_a_bram_matches_golden,
        test_stage_a_bram_matches_reg_on_many,
        test_dense_inner_matches_dot,
    ]:
        fn()
        print(fn.__name__, "OK")
