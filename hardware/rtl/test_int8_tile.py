"""Functional verification of INT8 RTL tiles vs NumPy integer golden model.

Amaranth built-in Simulator testbenches verifying:
1. StageAInt8Bram indexed accumulation and single-cycle bypass forwarding.
2. StageBInt8 integer reduction engine.
3. RailNetInt8Tile end-to-end inference matching golden arithmetic.
"""

import numpy as np
import pytest
from amaranth.sim import Simulator

from hardware.rtl.int8_tile import (
    RouteDecoderInt8,
    StageAInt8Bram,
    StageBInt8,
    RailNetInt8Tile,
    RAILS_DEFAULT,
)


def _golden_stage_a(x, routes_row, rails=RAILS_DEFAULT):
    g = np.zeros(rails, dtype=np.int64)
    for i, terms in enumerate(routes_row):
        for rail, sign in terms:
            val = int(x[i])
            g[rail] += -val if sign else val
    return g


def test_stage_a_int8_bram_functional():
    rails = 16
    dut = StageAInt8Bram(rails=rails)

    rng = np.random.default_rng(42)
    in_features = 24
    x = rng.integers(-100, 100, size=in_features, dtype=np.int16)

    # 1 term per input
    routes = []
    for i in range(in_features):
        r = int(rng.integers(0, rails))
        s = int(rng.integers(0, 2))
        routes.append([(r, s)])

    golden_g = _golden_stage_a(x, routes, rails=rails)
    got = np.zeros(rails, dtype=np.int64)

    async def tb(ctx):
        ctx.set(dut.flush, 1)
        await ctx.tick()
        ctx.set(dut.flush, 0)
        for _ in range(rails + 2):
            await ctx.tick()

        for i in range(in_features):
            r, s = routes[i][0]
            ctx.set(dut.x, int(x[i]))
            ctx.set(dut.rail, r)
            ctx.set(dut.sign, s)
            ctx.set(dut.valid, 1)
            await ctx.tick()

        ctx.set(dut.valid, 0)
        await ctx.tick()
        await ctx.tick()

        for r in range(rails):
            ctx.set(dut.g_addr, r)
            await ctx.tick()
            got[r] = ctx.get(dut.g_data)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    np.testing.assert_array_equal(got, golden_g)


def test_stage_a_int8_forwarding_hazard():
    """Test back-to-back writes to the EXACT same rail (forwarding bypass)."""
    rails = 8
    dut = StageAInt8Bram(rails=rails)

    target_rail = 3
    x_vals = [10, 20, -5, 15]
    expected_sum = 10 + 20 - 5 + 15  # 40

    got = {}

    async def tb(ctx):
        ctx.set(dut.flush, 1)
        await ctx.tick()
        ctx.set(dut.flush, 0)
        for _ in range(rails + 2):
            await ctx.tick()

        for val in x_vals:
            ctx.set(dut.x, val)
            ctx.set(dut.rail, target_rail)
            ctx.set(dut.sign, 0)
            ctx.set(dut.valid, 1)
            await ctx.tick()

        ctx.set(dut.valid, 0)
        await ctx.tick()
        await ctx.tick()

        ctx.set(dut.g_addr, target_rail)
        await ctx.tick()
        got["val"] = ctx.get(dut.g_data)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert got["val"] == expected_sum


def test_stage_b_int8_reduction():
    rails = 8
    dut = StageBInt8(rails=rails)

    rails_vals = np.array([1, 2, -4, 8, 16, -32, 64, -128], dtype=np.int8)
    g0 = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=np.int32)
    g1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    g2 = np.zeros(rails, dtype=np.int32)

    expected_y = int(np.sum((g0 + g1 + g2) * rails_vals))
    got = {}

    async def tb(ctx):
        # Program rails
        for r in range(rails):
            ctx.set(dut.rail_prog_en, 1)
            ctx.set(dut.rail_prog_addr, r)
            ctx.set(dut.rail_prog_data, int(rails_vals[r]))
            await ctx.tick()
        ctx.set(dut.rail_prog_en, 0)
        await ctx.tick()

        # Continuous G read-back provider
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        # Model 1-cycle synchronous SRAM read latency from Stage-A
        latched_addr = 0
        for _ in range(rails + 5):
            ctx.set(dut.g0_data, int(g0[latched_addr]))
            ctx.set(dut.g1_data, int(g1[latched_addr]))
            ctx.set(dut.g2_data, int(g2[latched_addr]))
            latched_addr = ctx.get(dut.g_addr)
            if ctx.get(dut.done):
                got["y"] = ctx.get(dut.y_out)
                break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert got["y"] == expected_y


def test_int8_tile_end_to_end():
    rails = 8
    dut = RailNetInt8Tile(rails=rails, codebook_depth=16)

    rails_vals = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int8)
    x_input = np.array([12, -7, 25, 3], dtype=np.int16)

    got = {}

    async def tb(ctx):
        # Program rails
        for r in range(rails):
            ctx.set(dut.prog_rail_en, 1)
            ctx.set(dut.prog_rail_addr, r)
            ctx.set(dut.prog_rail_data, int(rails_vals[r]))
            await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Program codebook: route 0 -> rail 1 (+), rail 2 (+)
        # term 0: rail=1, sign=0, active=1  -> bits [0..7]: 1 | (0<<7) | (1<<8) = 1 | 256 = 257
        # term 1: rail=2, sign=0, active=1  -> bits [8..16]: 2<<9 | 1<<17 = 1024 | 131072 = 132096
        entry0 = 257 | 132096
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, entry0)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # Flush tile
        ctx.set(dut.flush, 1)
        await ctx.tick()
        ctx.set(dut.flush, 0)
        for _ in range(rails + 2):
            await ctx.tick()

        # Feed activations with route 0
        for val in x_input:
            ctx.set(dut.x, int(val))
            ctx.set(dut.route_id, 0)
            ctx.set(dut.valid_in, 1)
            await ctx.tick()

        ctx.set(dut.valid_in, 0)
        await ctx.tick()
        await ctx.tick()

        # Start Stage-B reduction
        ctx.set(dut.start_reduction, 1)
        await ctx.tick()
        ctx.set(dut.start_reduction, 0)

        for _ in range(rails + 5):
            if ctx.get(dut.done):
                got["y"] = ctx.get(dut.y)
                break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Route 0 has terms: (rail 1, sign +), (rail 2, sign +)
    # Sum of x_input = 12 - 7 + 25 + 3 = 33
    # G[1] = 33, G[2] = 33
    # Y = G[1]*R[1] + G[2]*R[2] = 33*2 + 33*4 = 66 + 132 = 198
    assert got["y"] == 198
