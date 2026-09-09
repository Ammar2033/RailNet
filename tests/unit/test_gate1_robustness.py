"""Gate 1 Silicon Robustness Verification Suite for RailNet.

Validates the 5 critical showstopper fixes:
1. In-band CSR weight/route/codebook programming (Showstopper 1)
2. Multi-pass Stage-A chunked accumulation without auto-flush (Showstopper 2)
3. Sparse tile_mask concentrator without phantom duplicates (Showstopper 4)
4. Emergency global soft-reset abort across all FSM states (Showstopper 5)
5. 48-bit to 32-bit Stage-B symmetric saturation clamp (Architectural Risk 6)
"""

import pytest
import numpy as np
from amaranth.sim import Simulator

from hardware.rtl.axi import (
    REG_CTRL,
    REG_STATUS,
    REG_IN_FEATURES,
    REG_OUT_FEATURES,
    REG_TILE_MASK,
    REG_CYCLE_COUNT,
    REG_PROG_ADDR,
    REG_PROG_DATA,
    REG_PROG_CTRL,
)
from hardware.rtl.top import RailNetTop
from hardware.rtl.int8_tile import StageBInt8


async def _axi_write(ctx, dut, addr, data):
    """Execute standard 32-bit AXI4-Lite write transaction."""
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
    """Execute standard 32-bit AXI4-Lite read transaction."""
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


# ==============================================================================
# 1. Sparse Tile Mask & No Phantom Duplicates (Showstopper 4)
# ==============================================================================
def test_sparse_tile_mask_no_phantom_duplicates():
    """Verify that a sparse tile mask (e.g. 0b0101 in a 4-tile grid) emits ONLY
    the active tiles with exact word count and valid tlast framing.
    """
    num_tiles = 4
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    rails_vals = [1, 2, 4, 8, 16, 32, 64, -128]
    x_input = [10, 20]  # Sum = 30

    collected_y = []
    collected_last = []

    async def tb(ctx):
        # Program Tile 0: rail 1 (+2) -> weight = 2
        ctx.set(dut.prog_tile_idx, 0)
        for r in range(rails):
            ctx.set(dut.prog_rail_en, 1)
            ctx.set(dut.prog_rail_addr, r)
            ctx.set(dut.prog_rail_data, int(rails_vals[r]))
            await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Tile 0: route 0 has rail 1 (+2)
        # Term 0: rail=1, active=1 (bit 8) -> 1 | (1 << 8) = 257
        entry_tile0 = 257
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, entry_tile0)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        ctx.set(dut.prog_route_en, 1)
        for k in range(len(x_input)):
            ctx.set(dut.prog_route_addr, k)
            ctx.set(dut.prog_route_data, 0)
            await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # Program Tile 2: rail 0 (+1) and rail 1 (+2) -> weight = 3
        ctx.set(dut.prog_tile_idx, 2)
        for r in range(rails):
            ctx.set(dut.prog_rail_en, 1)
            ctx.set(dut.prog_rail_addr, r)
            ctx.set(dut.prog_rail_data, int(rails_vals[r]))
            await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Term 0: rail 0, active=1 -> 0 | 256 = 256
        # Term 1: rail 1, active=1 -> (1 << 9) | (1 << 17) = 512 | 131072 = 131584
        entry_tile2 = 256 | (1 << 9) | (1 << 17)
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, entry_tile2)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        ctx.set(dut.prog_route_en, 1)
        for k in range(len(x_input)):
            ctx.set(dut.prog_route_addr, k)
            ctx.set(dut.prog_route_data, 0)
            await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # Sparse mask: bit 0 and bit 2 active (0b0101 = 5)
        # Tile 1 and Tile 3 are disabled and must be skipped
        await _axi_write(ctx, dut, REG_TILE_MASK, 0b0101)
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

        # Monitor m_axis for exactly the active outputs
        for _ in range(rails + 40):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                collected_last.append(ctx.get(dut.m_axis_tlast))
                if len(collected_y) == 2:
                    break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Verify: exactly 2 outputs collected, no duplicates, correct tlast
    assert len(collected_y) == 2, f"Expected 2 words, got {len(collected_y)}: {collected_y}"
    assert collected_y[0] == 30 * 2  # Tile 0: 60
    assert collected_y[1] == 30 * 3  # Tile 2: 90
    assert collected_last == [0, 1], f"Expected [0, 1], got {collected_last}"


# ==============================================================================
# 2. Stage-B 48-bit to 32-bit Saturation Clamp (Architectural Risk 6)
# ==============================================================================
def test_stage_b_saturation_clamp():
    """Verify that StageBInt8 clamps positive and negative overflow symmetrically
    to 0x7FFFFFFF and -0x80000000 without 2's complement sign-flip wrapping.
    """
    rails = 4
    dut = StageBInt8(rails=rails)

    recorded_outputs = {}

    async def tb(ctx):
        # 1. Program Rail 0 to +100
        ctx.set(dut.rail_prog_en, 1)
        ctx.set(dut.rail_prog_addr, 0)
        ctx.set(dut.rail_prog_data, 100)
        await ctx.tick()

        # Program Rail 1 to -100
        ctx.set(dut.rail_prog_addr, 1)
        ctx.set(dut.rail_prog_data, -100)
        await ctx.tick()

        # Program Rail 2 and 3 to 0
        ctx.set(dut.rail_prog_addr, 2)
        ctx.set(dut.rail_prog_data, 0)
        await ctx.tick()
        ctx.set(dut.rail_prog_addr, 3)
        ctx.set(dut.rail_prog_data, 0)
        await ctx.tick()
        ctx.set(dut.rail_prog_en, 0)
        await ctx.tick()

        # --- Test Case A: Huge positive product causing > 0x7FFFFFFF accumulation ---
        # Provide g0 = 0x40000000 (1,073,741,824). Product with rail 0 (+100) is ~1.07e11 (48-bit)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        # Model 1-cycle synchronous SRAM read latency from Stage-A BRAM
        latched_addr = 0
        for _ in range(rails + 6):
            val = 0x40000000 if (latched_addr == 0) else 0
            ctx.set(dut.g0_data, val)
            ctx.set(dut.g1_data, 0)
            ctx.set(dut.g2_data, 0)
            latched_addr = ctx.get(dut.g_addr)
            if ctx.get(dut.done):
                recorded_outputs["pos_overflow"] = ctx.get(dut.y_out)
                break
            await ctx.tick()

        await ctx.tick()

        # --- Test Case B: Huge negative product causing < -0x80000000 accumulation ---
        # Provide g0 = 0x40000000, product with rail 1 (-100) is ~ -1.07e11
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        latched_addr = 0
        for _ in range(rails + 6):
            val = 0x40000000 if (latched_addr == 1) else 0
            ctx.set(dut.g0_data, val)
            ctx.set(dut.g1_data, 0)
            ctx.set(dut.g2_data, 0)
            latched_addr = ctx.get(dut.g_addr)
            if ctx.get(dut.done):
                recorded_outputs["neg_overflow"] = ctx.get(dut.y_out)
                break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Check saturation limits
    assert recorded_outputs["pos_overflow"] == 0x7FFFFFFF, (
        f"Expected 0x7FFFFFFF, got {hex(recorded_outputs['pos_overflow'])}"
    )
    assert recorded_outputs["neg_overflow"] == -0x80000000, (
        f"Expected -0x80000000, got {hex(recorded_outputs['neg_overflow'])}"
    )


# ==============================================================================
# 3. Emergency Global Soft Reset Abort (Showstopper 5)
# ==============================================================================
def test_global_soft_reset_recovery():
    """Verify that writing ctrl_soft_reset = 1 immediately aborts any active
    state (STREAM_IN, WAIT_RED, etc.) back to STATE_IDLE with status busy=0.
    """
    dut = RailNetTop(num_tiles=2, rails=8, codebook_depth=16, route_depth=16)

    status_before_reset = 0
    status_after_reset = 0

    async def tb(ctx):
        # Configure large in_features and start computation
        await _axi_write(ctx, dut, REG_IN_FEATURES, 100)
        await _axi_write(ctx, dut, REG_CTRL, 1)  # start=1

        # Wait until FSM enters busy state
        for _ in range(5):
            await ctx.tick()

        nonlocal status_before_reset
        status_before_reset = await _axi_read(ctx, dut, REG_STATUS)

        # Trigger emergency soft reset via CSR (bit 1 of REG_CTRL)
        await _axi_write(ctx, dut, REG_CTRL, 2)
        await ctx.tick()

        nonlocal status_after_reset
        status_after_reset = await _axi_read(ctx, dut, REG_STATUS)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Before reset, busy bit (bit 0) should be 1
    assert (status_before_reset & 0x1) == 1, "DUT failed to enter busy state"
    # After reset, busy bit (bit 0) must be 0 and done bit (bit 1) must be 0
    assert (status_after_reset & 0x1) == 0, "DUT remained busy after soft reset"
    assert (status_after_reset & 0x2) == 0, "DUT set done after soft reset"


# ==============================================================================
# 4. In-Band CSR Weight & Route Programming (Showstopper 1)
# ==============================================================================
def test_in_band_csr_programming():
    """Verify that tile weights, codebook entries, and route RAM can be programmed
    entirely in-band via AXI4-Lite CSR registers 0x18, 0x1C, 0x20.
    """
    num_tiles = 1
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    # In-band programming helper
    async def prog_csr(ctx, tile_idx, mem_type, addr, data):
        # mem_type: 0 = route, 1 = cb, 2 = rail
        prog_addr_val = (tile_idx & 0xFF) | ((mem_type & 0x3) << 8) | ((addr & 0xFFFF) << 16)
        await _axi_write(ctx, dut, REG_PROG_ADDR, prog_addr_val)
        await _axi_write(ctx, dut, REG_PROG_DATA, int(data))
        await _axi_write(ctx, dut, REG_PROG_CTRL, 1)  # strobe = 1

    x_input = [5, 15]  # Sum = 20
    collected_y = []

    async def tb(ctx):
        # Program Rail 2 = +7 via in-band CSR
        await prog_csr(ctx, tile_idx=0, mem_type=2, addr=2, data=7)

        # Program Codebook 0: rail 2 active -> (2 | 1 << 8) = 258
        entry_cb = 2 | (1 << 8)
        await prog_csr(ctx, tile_idx=0, mem_type=1, addr=0, data=entry_cb)

        # Program Route RAM: addresses 0 and 1 -> route 0
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=0, data=0)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=1, data=0)

        # Execute inference
        await _axi_write(ctx, dut, REG_IN_FEATURES, len(x_input))
        await _axi_write(ctx, dut, REG_TILE_MASK, 1)
        await _axi_write(ctx, dut, REG_CTRL, 1)  # start

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
                break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Sum(x) * 7 = 20 * 7 = 140
    assert len(collected_y) == 1, f"Expected 1 output, got {len(collected_y)}"
    assert collected_y[0] == 140, f"Expected 140, got {collected_y[0]}"


# ==============================================================================
# 5. Multi-Pass Stage-A Accumulation Without Flush (Showstopper 2)
# ==============================================================================
def test_multipass_chunking_accumulation():
    """Verify that streaming activations in two separate chunks using ctrl_accumulate=1
    preserves Stage-A accumulator sums and yields exact mathematical equivalence to single-pass.
    """
    num_tiles = 1
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    # Chunk 1: [10, 20] (Sum = 30)
    # Chunk 2: [5, 15]  (Sum = 20)
    # Combined Sum = 50. Rail 1 = +4 -> Final Expected Result = 50 * 4 = 200.
    collected_y = []

    async def tb(ctx):
        # 1. Program Rail 1 = 4
        ctx.set(dut.prog_tile_idx, 0)
        ctx.set(dut.prog_rail_en, 1)
        ctx.set(dut.prog_rail_addr, 1)
        ctx.set(dut.prog_rail_data, 4)
        await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Codebook entry: rail 1 active -> 1 | (1 << 8) = 257
        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, 257)
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # Route RAM addresses 0..3 -> route 0
        ctx.set(dut.prog_route_en, 1)
        for k in range(4):
            ctx.set(dut.prog_route_addr, k)
            ctx.set(dut.prog_route_data, 0)
            await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # === Pass 1: Chunk 1 with ctrl_accumulate = 0 (flushes accumulators) ===
        chunk1 = [10, 20]
        await _axi_write(ctx, dut, REG_IN_FEATURES, len(chunk1))
        await _axi_write(ctx, dut, REG_CTRL, 1)  # start=1, accumulate=0

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate(chunk1):
            is_last = 1 if (idx == len(chunk1) - 1) else 0
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, is_last)
            await ctx.tick()
        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        # Drain Pass 1 results (drain Stage-B output)
        ctx.set(dut.m_axis_tready, 1)
        while True:
            status = await _axi_read(ctx, dut, REG_STATUS)
            if (status & 0x2) != 0:  # done
                break
            await ctx.tick()

        # === Pass 2: Chunk 2 with ctrl_accumulate = 1 (DOES NOT flush accumulators) ===
        # CTRL register bit 0 = start, bit 2 = accumulate (value = 1 | 4 = 5)
        chunk2 = [5, 15]
        await _axi_write(ctx, dut, REG_IN_FEATURES, len(chunk2))
        await _axi_write(ctx, dut, REG_CTRL, 5)  # start=1, accumulate=1

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate(chunk2):
            is_last = 1 if (idx == len(chunk2) - 1) else 0
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, is_last)
            await ctx.tick()
        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        # Collect Pass 2 results
        for _ in range(rails + 20):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Pass 2 must reflect (10 + 20 + 5 + 15) * 4 = 50 * 4 = 200
    assert len(collected_y) == 1, f"Expected 1 output, got {len(collected_y)}"
    assert collected_y[0] == 200, f"Expected 200 from chunk accumulation, got {collected_y[0]}"
