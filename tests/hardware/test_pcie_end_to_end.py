"""End-to-End Cycle-Accurate Hardware Prototype Verification Suite.

Tests RailNet RTL hardware (RailNetTop and RailNetDualClockTop) directly
WITHOUT MockPCIeBridge, verifying:
1. End-to-end host DMA streaming to compute grid and result retrieval.
2. Exact bit-for-bit numerical contract against CPU golden model.
3. Downstream backpressure (m_axis_tready deassertion stall).
4. Upstream bubbles (s_axis_tvalid deassertion).
5. Soft-reset recovery during active computation.
6. Multi-tile scaling (2 and 4 tiles).
"""

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
from hardware.rtl.cdc import AxisCDCFIFO, AxiLiteCDCBridge, RailNetDualClockTop
from hardware.rtl.top import RailNetTop


async def axi_lite_write(ctx, dut, addr, data):
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


async def axi_lite_read(ctx, dut, addr):
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


# ---------------------------------------------------------------------------
# Test 1: Bit-Exact Numerical Contract (No Mock Bridge)
# ---------------------------------------------------------------------------
def test_hardware_e2e_bit_exact_numerical_contract():
    """Verify hardware grid against CPU golden dot product bit-for-bit."""
    num_tiles = 4
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=32)

    rails_vals = [2, -3, 5, -7, 11, -13, 17, -19]
    K = 16
    rng = np.random.default_rng(42)
    x_input = rng.integers(-50, 50, size=K, dtype=np.int16).tolist()

    # Define weights for 4 tiles:
    # Tile 0: W0 = R[0] + R[2] = 2 + 5 = 7
    # Tile 1: W1 = R[1] + R[3] = -3 - 7 = -10
    # Tile 2: W2 = R[4] + R[6] = 11 + 17 = 28
    # Tile 3: W3 = R[5] + R[7] = -13 - 19 = -32
    weights_per_tile = [7, -10, 28, -32]
    golden_outputs = [sum(x * w for x in x_input) for w in weights_per_tile]

    collected_y = []
    collected_last = []

    async def tb(ctx):
        # 1. Program Rails across all 4 tiles
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            for r in range(rails):
                ctx.set(dut.prog_rail_en, 1)
                ctx.set(dut.prog_rail_addr, r)
                ctx.set(dut.prog_rail_data, int(rails_vals[r]))
                await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # 2. Program Codebook entries
        # entry = rail0 | (rail1 << 9) | (sign0 << 8) | (sign1 << 17) | active flags
        # Tile 0: r0=0, r1=2
        cb_0 = 0 | (1 << 8) | (2 << 9) | (1 << 17)
        # Tile 1: r0=1, r1=3
        cb_1 = 1 | (1 << 8) | (3 << 9) | (1 << 17)
        # Tile 2: r0=4, r1=6
        cb_2 = 4 | (1 << 8) | (6 << 9) | (1 << 17)
        # Tile 3: r0=5, r1=7
        cb_3 = 5 | (1 << 8) | (7 << 9) | (1 << 17)

        cbs = [cb_0, cb_1, cb_2, cb_3]
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_cb_en, 1)
            ctx.set(dut.prog_cb_addr, 0)
            ctx.set(dut.prog_cb_data, cbs[t])
            await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # 3. Program Route Table: all addresses map to route 0
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_route_en, 1)
            for k in range(K):
                ctx.set(dut.prog_route_addr, k)
                ctx.set(dut.prog_route_data, 0)
                await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # 4. Configure CSR via AXI-Lite
        await axi_lite_write(ctx, dut, REG_IN_FEATURES, K)
        await axi_lite_write(ctx, dut, REG_OUT_FEATURES, num_tiles)
        await axi_lite_write(ctx, dut, REG_CTRL, 1)  # start

        # 5. Wait for s_axis_tready
        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        # 6. Stream activation vector via AXI-Stream
        for idx, val in enumerate(x_input):
            is_last = 1 if (idx == K - 1) else 0
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, is_last)
            await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        # 7. Accept outputs on m_axis
        ctx.set(dut.m_axis_tready, 1)
        for _ in range(rails + 30):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                collected_last.append(ctx.get(dut.m_axis_tlast))
                if len(collected_y) == num_tiles:
                    break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Exact bit-for-bit assertion
    assert len(collected_y) == num_tiles
    assert collected_y == golden_outputs
    assert collected_last == [0, 0, 0, 1]  # Only last tile asserts tlast


# ---------------------------------------------------------------------------
# Test 2: Downstream Backpressure & Stall Tolerance
# ---------------------------------------------------------------------------
def test_hardware_e2e_downstream_backpressure_stall():
    """Verify hardware stalls safely when m_axis_tready is deasserted."""
    num_tiles = 2
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    rails_vals = [3, 4, 5, 6, 7, 8, 9, 10]
    x_input = [10, 20, 30, 40]
    # Tile 0: W = R[0] = 3. Y0 = 100 * 3 = 300
    # Tile 1: W = R[1] = 4. Y1 = 100 * 4 = 400

    collected_y = []

    async def tb(ctx):
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            for r in range(rails):
                ctx.set(dut.prog_rail_en, 1)
                ctx.set(dut.prog_rail_addr, r)
                ctx.set(dut.prog_rail_data, int(rails_vals[r]))
                await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Codebook: t=0 -> r0=0; t=1 -> r0=1
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_cb_en, 1)
            ctx.set(dut.prog_cb_addr, 0)
            ctx.set(dut.prog_cb_data, t | (1 << 8))
            await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # Route table: map 0..3 to route 0
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_route_en, 1)
            for k in range(len(x_input)):
                ctx.set(dut.prog_route_addr, k)
                ctx.set(dut.prog_route_data, 0)
                await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        await axi_lite_write(ctx, dut, REG_IN_FEATURES, len(x_input))
        await axi_lite_write(ctx, dut, REG_CTRL, 1)

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate(x_input):
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, 1 if idx == len(x_input) - 1 else 0)
            await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)

        # Backpressure: hold m_axis_tready LOW initially
        ctx.set(dut.m_axis_tready, 0)

        # Wait until m_axis_tvalid is asserted
        while not ctx.get(dut.m_axis_tvalid):
            await ctx.tick()

        # Hold stall for 5 clock cycles
        for _ in range(5):
            assert ctx.get(dut.m_axis_tvalid) == 1, "m_axis_tvalid must stay high during stall"
            assert ctx.get(dut.m_axis_tdata) == 300, "m_axis_tdata must hold Tile 0 data (300) during stall"
            await ctx.tick()

        # Handshake beat 1: sample data during the valid cycle
        collected_y.append(ctx.get(dut.m_axis_tdata))
        ctx.set(dut.m_axis_tready, 1)
        await ctx.tick()

        # Deassert ready to stall beat 2
        ctx.set(dut.m_axis_tready, 0)
        for _ in range(3):
            assert ctx.get(dut.m_axis_tvalid) == 1, "m_axis_tvalid must stay high for beat 2 during stall"
            assert ctx.get(dut.m_axis_tdata) == 400, "m_axis_tdata must hold Tile 1 data (400) during stall"
            await ctx.tick()

        # Handshake beat 2: sample data
        collected_y.append(ctx.get(dut.m_axis_tdata))
        ctx.set(dut.m_axis_tready, 1)
        await ctx.tick()
        ctx.set(dut.m_axis_tready, 0)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert len(collected_y) == 2
    assert collected_y[0] == 300
    assert collected_y[1] == 400


# ---------------------------------------------------------------------------
# Test 3: Upstream Activation Bubbles Tolerance
# ---------------------------------------------------------------------------
def test_hardware_e2e_upstream_bubbles_tolerance():
    """Verify Stage-A handles non-consecutive activation stream bubbles."""
    num_tiles = 1
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    rails_vals = [5, 0, 0, 0, 0, 0, 0, 0]
    x_input = [10, 20, 30]  # Sum = 60 * 5 = 300

    collected_y = []

    async def tb(ctx):
        ctx.set(dut.prog_tile_idx, 0)
        ctx.set(dut.prog_rail_en, 1)
        ctx.set(dut.prog_rail_addr, 0)
        ctx.set(dut.prog_rail_data, 5)
        await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        ctx.set(dut.prog_cb_en, 1)
        ctx.set(dut.prog_cb_addr, 0)
        ctx.set(dut.prog_cb_data, 0 | (1 << 8))
        await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        ctx.set(dut.prog_route_en, 1)
        for k in range(3):
            ctx.set(dut.prog_route_addr, k)
            ctx.set(dut.prog_route_data, 0)
            await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        await axi_lite_write(ctx, dut, REG_IN_FEATURES, 3)
        await axi_lite_write(ctx, dut, REG_CTRL, 1)

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        # Send item 0
        ctx.set(dut.s_axis_tdata, 10)
        ctx.set(dut.s_axis_tvalid, 1)
        await ctx.tick()

        # Bubble 1: 2 cycles of tvalid = 0
        ctx.set(dut.s_axis_tvalid, 0)
        await ctx.tick()
        await ctx.tick()

        # Send item 1
        ctx.set(dut.s_axis_tdata, 20)
        ctx.set(dut.s_axis_tvalid, 1)
        await ctx.tick()

        # Bubble 2: 1 cycle of tvalid = 0
        ctx.set(dut.s_axis_tvalid, 0)
        await ctx.tick()

        # Send item 2 (last)
        ctx.set(dut.s_axis_tdata, 30)
        ctx.set(dut.s_axis_tvalid, 1)
        ctx.set(dut.s_axis_tlast, 1)
        await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        ctx.set(dut.m_axis_tready, 1)
        while not ctx.get(dut.m_axis_tvalid):
            await ctx.tick()

        collected_y.append(ctx.get(dut.m_axis_tdata))

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert len(collected_y) == 1
    assert collected_y[0] == 300


# ---------------------------------------------------------------------------
# Test 4: Soft-Reset Recovery During Active Processing
# ---------------------------------------------------------------------------
def test_hardware_e2e_soft_reset_recovery():
    """Verify soft reset brings FSM cleanly back to IDLE state."""
    dut = RailNetTop(num_tiles=2, rails=8, codebook_depth=16, route_depth=16)

    status_before_reset = None
    status_after_reset = None

    async def tb(ctx):
        # Configure and start
        await axi_lite_write(ctx, dut, REG_IN_FEATURES, 100)
        await axi_lite_write(ctx, dut, REG_CTRL, 1)  # start

        # Wait until FSM enters BUSY state
        await ctx.tick()
        await ctx.tick()
        nonlocal status_before_reset
        status_before_reset = await axi_lite_read(ctx, dut, REG_STATUS)

        # Issue soft-reset: CTRL[soft_reset] = 1 (bit 1 -> 0x2)
        await axi_lite_write(ctx, dut, REG_CTRL, 0x2)
        await ctx.tick()
        await ctx.tick()

        nonlocal status_after_reset
        status_after_reset = await axi_lite_read(ctx, dut, REG_STATUS)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Bit 0 of STATUS is 'busy'
    assert status_after_reset & 0x1 == 0, "FSM must return to IDLE (busy=0) after soft-reset"


# ---------------------------------------------------------------------------
# Test 5: Varying Tensor Dimensions (K = 8, 16, 32) Across Multiple Tiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [8, 16, 32])
def test_hardware_e2e_varying_tensor_dimensions(K):
    """Verify hardware compute grid across varying vector dimensions K."""
    num_tiles = 2
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=K + 8)

    rails_vals = [3, -5, 7, -11, 13, -17, 19, -23]
    rng = np.random.default_rng(K * 7)
    x_input = rng.integers(-40, 40, size=K, dtype=np.int16).tolist()

    # Tile 0: W0 = R[0] + R[2] = 3 + 7 = 10
    # Tile 1: W1 = R[1] + R[3] = -5 - 11 = -16
    w0 = 10
    w1 = -16
    golden_outputs = [sum(x * w0 for x in x_input), sum(x * w1 for x in x_input)]

    collected_y = []

    async def tb(ctx):
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            for r in range(rails):
                ctx.set(dut.prog_rail_en, 1)
                ctx.set(dut.prog_rail_addr, r)
                ctx.set(dut.prog_rail_data, int(rails_vals[r]))
                await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        # Codebook:
        # Tile 0: r0=0, r1=2
        # Tile 1: r0=1, r1=3
        cb0 = 0 | (1 << 8) | (2 << 9) | (1 << 17)
        cb1 = 1 | (1 << 8) | (3 << 9) | (1 << 17)
        for t, cb_val in enumerate([cb0, cb1]):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_cb_en, 1)
            ctx.set(dut.prog_cb_addr, 0)
            ctx.set(dut.prog_cb_data, cb_val)
            await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        # Route table
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_route_en, 1)
            for k in range(K):
                ctx.set(dut.prog_route_addr, k)
                ctx.set(dut.prog_route_data, 0)
                await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        await axi_lite_write(ctx, dut, REG_IN_FEATURES, K)
        await axi_lite_write(ctx, dut, REG_OUT_FEATURES, num_tiles)
        await axi_lite_write(ctx, dut, REG_CTRL, 1)

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate(x_input):
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, 1 if idx == K - 1 else 0)
            await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        ctx.set(dut.m_axis_tready, 1)
        for _ in range(rails + 40):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                if len(collected_y) == num_tiles:
                    break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert len(collected_y) == num_tiles, f"Expected {num_tiles} outputs for K={K}, got {len(collected_y)}"
    assert collected_y == golden_outputs, f"Mismatch for K={K}: {collected_y} != {golden_outputs}"


# ---------------------------------------------------------------------------
# Test 6: Interleaved Simultaneous Upstream Bubbles & Downstream Backpressure
# ---------------------------------------------------------------------------
def test_hardware_e2e_interleaved_backpressure_and_bubbles():
    """Verify robust execution when both upstream and downstream stall cleanly."""
    num_tiles = 2
    rails = 8
    K = 16
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=32)

    rails_vals = [4, 6, -2, -3, 5, 7, -1, -4]
    rng = np.random.default_rng(2026)
    x_input = rng.integers(-20, 20, size=K, dtype=np.int16).tolist()

    # Tile 0: R[0] = 4
    # Tile 1: R[1] = 6
    golden_outputs = [sum(x * 4 for x in x_input), sum(x * 6 for x in x_input)]
    collected_y = []

    async def tb(ctx):
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            for r in range(rails):
                ctx.set(dut.prog_rail_en, 1)
                ctx.set(dut.prog_rail_addr, r)
                ctx.set(dut.prog_rail_data, int(rails_vals[r]))
                await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_cb_en, 1)
            ctx.set(dut.prog_cb_addr, 0)
            ctx.set(dut.prog_cb_data, t | (1 << 8))
            await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_route_en, 1)
            for k in range(K):
                ctx.set(dut.prog_route_addr, k)
                ctx.set(dut.prog_route_data, 0)
                await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        await axi_lite_write(ctx, dut, REG_IN_FEATURES, K)
        await axi_lite_write(ctx, dut, REG_OUT_FEATURES, num_tiles)
        await axi_lite_write(ctx, dut, REG_CTRL, 1)

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        # Stream activations with deterministic bubbles every 2 beats
        for idx, val in enumerate(x_input):
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, 1 if idx == K - 1 else 0)
            await ctx.tick()

            # Bubble: insert 2 cycles of tvalid = 0 between beats
            if idx % 3 == 0 and idx < K - 1:
                ctx.set(dut.s_axis_tvalid, 0)
                await ctx.tick()
                await ctx.tick()

        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        # Receive outputs with deliberate downstream backpressure stalls
        ctx.set(dut.m_axis_tready, 0)
        while not ctx.get(dut.m_axis_tvalid):
            await ctx.tick()

        # Stall tile 0 for 4 cycles
        for _ in range(4):
            assert ctx.get(dut.m_axis_tvalid) == 1
            await ctx.tick()

        collected_y.append(ctx.get(dut.m_axis_tdata))
        ctx.set(dut.m_axis_tready, 1)
        await ctx.tick()

        # Stall tile 1 for 3 cycles
        ctx.set(dut.m_axis_tready, 0)
        for _ in range(3):
            assert ctx.get(dut.m_axis_tvalid) == 1
            await ctx.tick()

        collected_y.append(ctx.get(dut.m_axis_tdata))
        ctx.set(dut.m_axis_tready, 1)
        await ctx.tick()
        ctx.set(dut.m_axis_tready, 0)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert len(collected_y) == num_tiles
    assert collected_y == golden_outputs


# ---------------------------------------------------------------------------
# Test 7: Sustained Multi-Token Stability Stress Test
# ---------------------------------------------------------------------------
def test_hardware_e2e_sustained_multitoken_stress():
    """Verify hardware runs multiple consecutive inference tokens stably."""
    num_tiles = 2
    rails = 8
    K = 16
    NUM_TOKENS = 5
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=32)

    rails_vals = [2, 3, 0, 0, 0, 0, 0, 0]
    rng = np.random.default_rng(999)

    token_inputs = [rng.integers(-10, 10, size=K, dtype=np.int16).tolist() for _ in range(NUM_TOKENS)]
    expected_outputs = []
    for x in token_inputs:
        expected_outputs.append([sum(val * 2 for val in x), sum(val * 3 for val in x)])

    all_received_outputs = []

    async def tb(ctx):
        # 1. Program static rails and routing once
        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            for r in range(rails):
                ctx.set(dut.prog_rail_en, 1)
                ctx.set(dut.prog_rail_addr, r)
                ctx.set(dut.prog_rail_data, int(rails_vals[r]))
                await ctx.tick()
        ctx.set(dut.prog_rail_en, 0)

        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_cb_en, 1)
            ctx.set(dut.prog_cb_addr, 0)
            ctx.set(dut.prog_cb_data, t | (1 << 8))
            await ctx.tick()
        ctx.set(dut.prog_cb_en, 0)

        for t in range(num_tiles):
            ctx.set(dut.prog_tile_idx, t)
            ctx.set(dut.prog_route_en, 1)
            for k in range(K):
                ctx.set(dut.prog_route_addr, k)
                ctx.set(dut.prog_route_data, 0)
                await ctx.tick()
        ctx.set(dut.prog_route_en, 0)

        # 2. Stream tokens sequentially
        for token_idx in range(NUM_TOKENS):
            x_input = token_inputs[token_idx]
            await axi_lite_write(ctx, dut, REG_IN_FEATURES, K)
            await axi_lite_write(ctx, dut, REG_OUT_FEATURES, num_tiles)
            await axi_lite_write(ctx, dut, REG_CTRL, 1)  # start token

            while not ctx.get(dut.s_axis_tready):
                await ctx.tick()

            for idx, val in enumerate(x_input):
                ctx.set(dut.s_axis_tdata, val & 0xFFFF)
                ctx.set(dut.s_axis_tvalid, 1)
                ctx.set(dut.s_axis_tlast, 1 if idx == K - 1 else 0)
                await ctx.tick()

            ctx.set(dut.s_axis_tvalid, 0)
            ctx.set(dut.s_axis_tlast, 0)

            token_y = []
            ctx.set(dut.m_axis_tready, 1)
            timeout = 0
            while len(token_y) < num_tiles and timeout < 200:
                if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                    token_y.append(ctx.get(dut.m_axis_tdata))
                await ctx.tick()
                timeout += 1

            ctx.set(dut.m_axis_tready, 0)
            all_received_outputs.append(token_y)

            # Allow FSM to cleanly settle into STATE_DONE
            for _ in range(5):
                await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert len(all_received_outputs) == NUM_TOKENS
    for t_idx in range(NUM_TOKENS):
        assert all_received_outputs[t_idx] == expected_outputs[t_idx], f"Token {t_idx} output mismatch"


# ---------------------------------------------------------------------------
# Test 8: Dual-Clock CDC Ratio (Host 50 MHz vs Core 100 MHz CSR Crossing)
# ---------------------------------------------------------------------------
def test_hardware_e2e_dual_clock_cdc_ratio():
    """Verify RailNetDualClockTop with asynchronous 2:1 host-to-core clock ratio."""
    num_tiles = 2
    rails = 8
    dut = RailNetDualClockTop(
        num_tiles=num_tiles,
        rails=rails,
        codebook_depth=16,
        route_depth=16,
        fifo_depth=8,
    )

    HOST_PERIOD = 20e-9  # 50 MHz
    CORE_PERIOD = 10e-9  # 100 MHz

    readbacks = {}

    async def tb_host(ctx):
        for _ in range(20):
            await ctx.tick("host")

        # AXI-Lite write via CDC bridge: set in_features = 64
        ctx.set(dut.s_axi_awaddr, REG_IN_FEATURES)
        ctx.set(dut.s_axi_awvalid, 1)
        ctx.set(dut.s_axi_wdata, 64)
        ctx.set(dut.s_axi_wstrb, 0xF)
        ctx.set(dut.s_axi_wvalid, 1)
        ctx.set(dut.s_axi_bready, 1)
        for _ in range(200):
            if ctx.get(dut.s_axi_awready) and ctx.get(dut.s_axi_wready):
                break
            await ctx.tick("host")
        await ctx.tick("host")
        ctx.set(dut.s_axi_awvalid, 0)
        ctx.set(dut.s_axi_wvalid, 0)
        for _ in range(200):
            if ctx.get(dut.s_axi_bvalid):
                break
            await ctx.tick("host")
        await ctx.tick("host")
        ctx.set(dut.s_axi_bready, 0)

        # Write TILE_MASK = 0b11
        ctx.set(dut.s_axi_awaddr, REG_TILE_MASK)
        ctx.set(dut.s_axi_awvalid, 1)
        ctx.set(dut.s_axi_wdata, 0b11)
        ctx.set(dut.s_axi_wstrb, 0xF)
        ctx.set(dut.s_axi_wvalid, 1)
        ctx.set(dut.s_axi_bready, 1)
        for _ in range(200):
            if ctx.get(dut.s_axi_awready) and ctx.get(dut.s_axi_wready):
                break
            await ctx.tick("host")
        await ctx.tick("host")
        ctx.set(dut.s_axi_awvalid, 0)
        ctx.set(dut.s_axi_wvalid, 0)
        for _ in range(200):
            if ctx.get(dut.s_axi_bvalid):
                break
            await ctx.tick("host")
        await ctx.tick("host")
        ctx.set(dut.s_axi_bready, 0)

        # Read back IN_FEATURES
        ctx.set(dut.s_axi_araddr, REG_IN_FEATURES)
        ctx.set(dut.s_axi_arvalid, 1)
        ctx.set(dut.s_axi_rready, 1)
        for _ in range(200):
            if ctx.get(dut.s_axi_arready):
                break
            await ctx.tick("host")
        await ctx.tick("host")
        ctx.set(dut.s_axi_arvalid, 0)
        for _ in range(200):
            if ctx.get(dut.s_axi_rvalid):
                break
            await ctx.tick("host")
        readbacks["in_feat"] = ctx.get(dut.s_axi_rdata)
        await ctx.tick("host")
        ctx.set(dut.s_axi_rready, 0)

        # Read back TILE_MASK
        ctx.set(dut.s_axi_araddr, REG_TILE_MASK)
        ctx.set(dut.s_axi_arvalid, 1)
        ctx.set(dut.s_axi_rready, 1)
        for _ in range(200):
            if ctx.get(dut.s_axi_arready):
                break
            await ctx.tick("host")
        await ctx.tick("host")
        ctx.set(dut.s_axi_arvalid, 0)
        for _ in range(200):
            if ctx.get(dut.s_axi_rvalid):
                break
            await ctx.tick("host")
        readbacks["tile_mask"] = ctx.get(dut.s_axi_rdata)
        await ctx.tick("host")
        ctx.set(dut.s_axi_rready, 0)

    sim = Simulator(dut)
    sim.add_clock(HOST_PERIOD, domain="host")
    sim.add_clock(CORE_PERIOD, domain="core")
    sim.add_testbench(tb_host)
    sim.run()

    assert readbacks["in_feat"] == 64, f"Expected IN_FEATURES=64, got {readbacks.get('in_feat')}"
    assert readbacks["tile_mask"] == 0b11, f"Expected TILE_MASK=0b11, got {readbacks.get('tile_mask')}"


