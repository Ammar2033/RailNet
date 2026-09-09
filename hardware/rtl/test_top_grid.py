"""Functional verification of RailNetTop multi-tile grid with AXI-Lite and AXI-Stream."""

import numpy as np
import pytest
from amaranth.sim import Simulator

from hardware.rtl.axi import (
    REG_CTRL,
    REG_CYCLE_COUNT,
    REG_IN_FEATURES,
    REG_OUT_FEATURES,
    REG_STATUS,
    REG_TILE_MASK,
)
from hardware.rtl.top import RailNetTop


async def _axi_write(ctx, dut, addr, data):
    """Perform a standard 32-bit AXI4-Lite write transaction."""
    ctx.set(dut.s_axi_awaddr, addr)
    ctx.set(dut.s_axi_awvalid, 1)
    ctx.set(dut.s_axi_wdata, data)
    ctx.set(dut.s_axi_wstrb, 0xF)
    ctx.set(dut.s_axi_wvalid, 1)
    ctx.set(dut.s_axi_bready, 1)

    while not (ctx.get(dut.s_axi_awready) and ctx.get(dut.s_axi_wready)):
        await ctx.tick()

    await ctx.tick()
    ctx.set(dut.s_axi_awvalid, 0)
    ctx.set(dut.s_axi_wvalid, 0)

    while not ctx.get(dut.s_axi_bvalid):
        await ctx.tick()
    await ctx.tick()
    ctx.set(dut.s_axi_bready, 0)


async def _axi_read(ctx, dut, addr):
    """Perform a standard 32-bit AXI4-Lite read transaction."""
    ctx.set(dut.s_axi_araddr, addr)
    ctx.set(dut.s_axi_arvalid, 1)
    ctx.set(dut.s_axi_rready, 1)

    while not ctx.get(dut.s_axi_arready):
        await ctx.tick()

    await ctx.tick()
    ctx.set(dut.s_axi_arvalid, 0)

    while not ctx.get(dut.s_axi_rvalid):
        await ctx.tick()
    val = ctx.get(dut.s_axi_rdata)
    await ctx.tick()
    ctx.set(dut.s_axi_rready, 0)
    return val


def test_axi_csr_registers():
    dut = RailNetTop(num_tiles=4, rails=8, codebook_depth=16, route_depth=16)

    readbacks = {}

    async def tb(ctx):
        # Initial status read
        readbacks["status_init"] = await _axi_read(ctx, dut, REG_STATUS)

        # Write and read IN_FEATURES
        await _axi_write(ctx, dut, REG_IN_FEATURES, 256)
        readbacks["in_feat"] = await _axi_read(ctx, dut, REG_IN_FEATURES)

        # Write and read OUT_FEATURES
        await _axi_write(ctx, dut, REG_OUT_FEATURES, 16)
        readbacks["out_feat"] = await _axi_read(ctx, dut, REG_OUT_FEATURES)

        # Write and read TILE_MASK
        await _axi_write(ctx, dut, REG_TILE_MASK, 0b1011)
        readbacks["tile_mask"] = await _axi_read(ctx, dut, REG_TILE_MASK)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Verify CSR readbacks
    assert readbacks["in_feat"] == 256
    assert readbacks["out_feat"] == 16
    assert readbacks["tile_mask"] == 0b1011
    # Check num_tiles = 4 encoded in bits [15:8] of STATUS
    assert (readbacks["status_init"] >> 8) & 0xFF == 4


def test_top_grid_e2e_parallel_inference():
    """End-to-end multi-tile parallel inference with AXI-Lite & AXI-Stream."""
    num_tiles = 2
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    rails_vals = [1, 2, 4, 8, 16, 32, 64, -128]
    # Inputs: 4 elements -> Sum = 10 + 20 - 5 + 15 = 40
    x_input = [10, 20, -5, 15]

    collected_y = []
    collected_last = []
    cycles_recorded = 0

    async def tb(ctx):
        # 1. Program Rail values for both tiles
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            for r in range(rails):
                ctx.set(dut.prog_rail_en, 1)
                ctx.set(dut.prog_rail_addr, r)
                ctx.set(dut.prog_rail_data, int(rails_vals[r]))
                await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # 2. Program Codebook
        # Tile 0 route 0: rail 1 (+2) and rail 2 (+4) -> weight = 6
        entry_tile0 = 257 | 132096  # (rail 1 active, rail 2 active)
        ctx.set(dut.prog_tile_idx, 0)
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, entry_tile0)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # Tile 1 route 0: rail 0 (+1) and rail 3 (+8) -> weight = 9
        # term 0: rail=0, active=1 -> bits [0..7]: 1<<8 = 256
        # term 1: rail=3, active=1 -> bits [8..16]: 3<<9 | 1<<17 = 1536 | 131072 = 132608
        entry_tile1 = 256 | 132608
        ctx.set(dut.prog_tile_idx, 1)
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, entry_tile1)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # 3. Program Route RAMs: addresses 0..3 have route 0
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_route_en, 1)
            for k in range(len(x_input)):
                ctx.set(dut.prog_route_addr, k)
                ctx.set(dut.prog_route_data, 0)
                await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # 4. Configure CSR via AXI-Lite: in_features = 4
        await _axi_write(ctx, dut, REG_IN_FEATURES, len(x_input))

        # 5. Start computation via CSR: CTRL[start] = 1
        await _axi_write(ctx, dut, REG_CTRL, 1)

        # 6. Wait for s_axis_tready = 1 (FSM leaves flush state)
        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        # 7. Stream activation inputs over AXI4-Stream
        for idx, val in enumerate(x_input):
            is_last = 1 if (idx == len(x_input) - 1) else 0
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, is_last)
            await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        # 8. Accept results from m_axis
        ctx.set(dut.m_axis_tready, 1)

        # Wait for outputs
        for _ in range(rails + 20):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                collected_last.append(ctx.get(dut.m_axis_tlast))
                if len(collected_y) == num_tiles:
                    break
            await ctx.tick()

        nonlocal cycles_recorded
        cycles_recorded = await _axi_read(ctx, dut, REG_CYCLE_COUNT)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Golden verification:
    # Sum(X) = 10 + 20 - 5 + 15 = 40
    # Tile 0: 40 * (R[1] + R[2]) = 40 * (2 + 4) = 40 * 6 = 240
    # Tile 1: 40 * (R[0] + R[3]) = 40 * (1 + 8) = 40 * 9 = 360
    assert len(collected_y) == 2
    assert collected_y[0] == 240
    assert collected_y[1] == 360
    assert collected_last == [0, 1]  # Tile 1 is the last active tile
    assert cycles_recorded > 0


def test_top_grid_tile_masking():
    """Verify that disabled tiles in tile_mask are skipped during collection."""
    num_tiles = 2
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    rails_vals = [1, 2, 4, 8, 16, 32, 64, -128]
    x_input = [10, 20]

    collected_y = []
    collected_last = []

    async def tb(ctx):
        # Program Tile 1 only
        ctx.set(dut.prog_tile_idx, 1)
        for r in range(rails):
            ctx.set(dut.prog_rail_en, 1)
            ctx.set(dut.prog_rail_addr, r)
            ctx.set(dut.prog_rail_data, int(rails_vals[r]))
            await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Tile 1 route 0: rail 0 (+1) and rail 3 (+8) -> weight = 9
        entry_tile1 = 256 | 132608
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, entry_tile1)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        ctx.set(dut.prog_route_en, 1)
        for k in range(len(x_input)):
            ctx.set(dut.prog_route_addr, k)
            ctx.set(dut.prog_route_data, 0)
            await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # Set TILE_MASK = 0b0010 (Only Tile 1 active)
        await _axi_write(ctx, dut, REG_TILE_MASK, 0b0010)
        await _axi_write(ctx, dut, REG_IN_FEATURES, len(x_input))
        await _axi_write(ctx, dut, REG_CTRL, 1)

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate(x_input):
            is_last = 1 if (idx == len(x_input) - 1) else 0
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, is_last)
            await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        ctx.set(dut.m_axis_tready, 1)
        for _ in range(rails + 20):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                collected_last.append(ctx.get(dut.m_axis_tlast))
                if len(collected_y) == 1:
                    break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Only Tile 1 was collected: (10 + 20) * 9 = 270
    assert len(collected_y) == 1
    assert collected_y[0] == 270
    assert collected_last[0] == 1  # Since it's the only active tile, last is 1

