"""Gate 2 Verification Suite: Ping-Pong Double-Buffering & Prefetching.

Validates:
1. Bank isolation: Writing background bank (~active_bank) does not perturb active compute.
2. Autonomous hardware bank swap on STATE_DONE when arm_next_bank=1 and auto_swap_en=1.
3. Manual bank swap via REG_CTRL[manual_swap].
4. PCIe driver prefetch_layer and double-buffering integration.
5. Asynchronous background prefetch worker thread.
"""

import time
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
from railnet.runtime.pcie import RailNetPCIeDriver, MockPCIeBridge
from railnet.kernel import CompiledTensor, prepare, rail_linear_fast


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


def _make_dummy_compiled(
    out_features: int = 4,
    in_features: int = 8,
    rail_count: int = 8,
    max_terms: int = 2,
    num_routes: int = 16,
    seed: int = 42,
) -> CompiledTensor:
    rng = np.random.RandomState(seed)
    c = CompiledTensor.__new__(CompiledTensor)
    c.checksum_ok = True
    c.tensor_name = "gate2_dummy"
    c.rail_count = rail_count
    c.max_terms = max_terms
    c.shape = (out_features, in_features)
    c.out_features = out_features
    c.in_features = in_features
    c.load_seconds = 0.0
    c.scale = 1.0

    c.rails_f64 = rng.uniform(-0.5, 0.5, size=rail_count).astype(np.float64)
    c.term_rail = np.zeros((65536, max_terms), dtype=np.int32)
    c.term_sign = np.zeros((65536, max_terms), dtype=np.int8)
    c.term_active = np.zeros((65536, max_terms), dtype=bool)

    for g in range(num_routes):
        num_t = rng.randint(1, max_terms + 1)
        r_indices = rng.choice(rail_count, size=num_t, replace=False)
        signs = rng.choice([-1, 1], size=num_t)
        for t in range(num_t):
            c.term_rail[g, t] = r_indices[t]
            c.term_sign[g, t] = signs[t]
            c.term_active[g, t] = True

    c.route_ids = rng.randint(0, num_routes, size=(out_features, in_features)).astype(np.int32)
    c.prepared = False
    prepare(c)
    return c


# ==============================================================================
# 1. RTL Bank Isolation Test
# ==============================================================================
def test_bank_isolation():
    """Verify that Bank 0 compute is strictly isolated from Bank 1 writes."""
    num_tiles = 1
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    collected_y = []

    async def prog_csr(ctx, tile_idx, mem_type, addr, data):
        prog_addr_val = (tile_idx & 0xFF) | ((mem_type & 0x3) << 8) | ((addr & 0xFFFF) << 16)
        await _axi_write(ctx, dut, REG_PROG_ADDR, prog_addr_val)
        await _axi_write(ctx, dut, REG_PROG_DATA, int(data))
        await _axi_write(ctx, dut, REG_PROG_CTRL, 1)

    async def tb(ctx):
        # 1. Program Bank 0 (auto_swap_en = 0 -> prog_bank = active_bank = 0)
        # Rail 1 = +3, Codebook 0: rail 1 active -> 1 | (1 << 8) = 257
        await prog_csr(ctx, tile_idx=0, mem_type=2, addr=1, data=3)
        await prog_csr(ctx, tile_idx=0, mem_type=1, addr=0, data=257)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=0, data=0)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=1, data=0)

        # 2. Enable auto_swap_en = 1 (bit 5 = 32). Now prog_bank points to Bank 1 (~active_bank)
        await _axi_write(ctx, dut, REG_CTRL, 32)

        # 3. Program Bank 1 with completely different values: Rail 1 = +99
        await prog_csr(ctx, tile_idx=0, mem_type=2, addr=1, data=99)
        await prog_csr(ctx, tile_idx=0, mem_type=1, addr=0, data=257)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=0, data=0)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=1, data=0)

        # 4. Execute inference on Bank 0 (x = [10, 20], sum = 30)
        # Should use Bank 0 values (Rail 1 = +3 -> Output = 30 * 3 = 90)
        await _axi_write(ctx, dut, REG_IN_FEATURES, 2)
        await _axi_write(ctx, dut, REG_TILE_MASK, 1)
        # start=1, auto_swap_en=1 (value = 1 | 32 = 33)
        await _axi_write(ctx, dut, REG_CTRL, 33)

        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate([10, 20]):
            is_last = 1 if (idx == 1) else 0
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

    # Bank 0 was computed: 30 * 3 = 90 (NOT 30 * 99 = 2970)
    assert len(collected_y) == 1
    assert collected_y[0] == 90, f"Expected 90 from Bank 0, got {collected_y[0]}"


# ==============================================================================
# 2. RTL Autonomous Bank Swap Test
# ==============================================================================
def test_autonomous_bank_swap():
    """Verify autonomous zero-bubble bank swap on STATE_DONE when armed."""
    num_tiles = 1
    rails = 8
    dut = RailNetTop(num_tiles=num_tiles, rails=rails, codebook_depth=16, route_depth=16)

    collected_y = []
    status_history = []

    async def prog_csr(ctx, tile_idx, mem_type, addr, data):
        prog_addr_val = (tile_idx & 0xFF) | ((mem_type & 0x3) << 8) | ((addr & 0xFFFF) << 16)
        await _axi_write(ctx, dut, REG_PROG_ADDR, prog_addr_val)
        await _axi_write(ctx, dut, REG_PROG_DATA, int(data))
        await _axi_write(ctx, dut, REG_PROG_CTRL, 1)

    async def tb(ctx):
        # 1. Program Bank 0 (Rail 1 = +2) -> weight = 2
        await prog_csr(ctx, tile_idx=0, mem_type=2, addr=1, data=2)
        await prog_csr(ctx, tile_idx=0, mem_type=1, addr=0, data=257)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=0, data=0)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=1, data=0)

        # 2. Enable auto_swap_en (bit 5 = 32)
        await _axi_write(ctx, dut, REG_CTRL, 32)

        # 3. Preload Bank 1 (Rail 1 = +5) -> weight = 5
        await prog_csr(ctx, tile_idx=0, mem_type=2, addr=1, data=5)
        await prog_csr(ctx, tile_idx=0, mem_type=1, addr=0, data=257)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=0, data=0)
        await prog_csr(ctx, tile_idx=0, mem_type=0, addr=1, data=0)

        # 4. Arm the swap: bit 0 (start), bit 4 (arm_next_bank), bit 5 (auto_swap_en)
        # Value = 1 | 16 | 32 = 49
        await _axi_write(ctx, dut, REG_IN_FEATURES, 2)
        await _axi_write(ctx, dut, REG_TILE_MASK, 1)
        await _axi_write(ctx, dut, REG_CTRL, 49)

        # Execute Pass 1 (Bank 0): input [10, 20] -> Sum = 30 * 2 = 60
        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        for idx, val in enumerate([10, 20]):
            is_last = 1 if (idx == 1) else 0
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

        # Check status: active_bank should have autonomously swapped to 1!
        # Give 2 cycles for STATE_DONE auto-swap to trigger
        await ctx.tick()
        await ctx.tick()
        status_history.append(await _axi_read(ctx, dut, REG_STATUS))

        # Pass 2 is now running on Bank 1! Wait for s_axis_tready
        while not ctx.get(dut.s_axis_tready):
            await ctx.tick()

        # Stream Pass 2 inputs on Bank 1: input [10, 20] -> Sum = 30 * 5 = 150
        for idx, val in enumerate([10, 20]):
            is_last = 1 if (idx == 1) else 0
            ctx.set(dut.s_axis_tdata, val & 0xFFFF)
            ctx.set(dut.s_axis_tvalid, 1)
            ctx.set(dut.s_axis_tlast, is_last)
            await ctx.tick()
        ctx.set(dut.s_axis_tvalid, 0)
        ctx.set(dut.s_axis_tlast, 0)

        for _ in range(rails + 20):
            if ctx.get(dut.m_axis_tvalid) and ctx.get(dut.m_axis_tready):
                collected_y.append(ctx.get(dut.m_axis_tdata))
                break
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    # Pass 1: Bank 0 (30 * 2 = 60)
    # Pass 2: Bank 1 (30 * 5 = 150)
    assert len(collected_y) == 2, f"Expected 2 outputs, got {len(collected_y)}: {collected_y}"
    assert collected_y[0] == 60
    assert collected_y[1] == 150
    # Status check: bit 2 of STATUS is active_bank (should be 1)
    status_val = status_history[0]
    active_bank = (status_val >> 2) & 0x1
    assert active_bank == 1, f"Expected active_bank == 1, got {active_bank} in status {hex(status_val)}"


# ==============================================================================
# 3. Manual Bank Swap Test
# ==============================================================================
def test_manual_bank_swap():
    """Verify manual bank swap via REG_CTRL[manual_swap]."""
    dut = RailNetTop(num_tiles=2, rails=8, codebook_depth=16, route_depth=16)

    bank_reads = []

    async def tb(ctx):
        # Initial status: active_bank should be 0
        s0 = await _axi_read(ctx, dut, REG_STATUS)
        bank_reads.append((s0 >> 2) & 0x1)

        # Pulse manual_swap (bit 6 = 64)
        await _axi_write(ctx, dut, REG_CTRL, 64)
        await ctx.tick()

        s1 = await _axi_read(ctx, dut, REG_STATUS)
        bank_reads.append((s1 >> 2) & 0x1)

        # Pulse manual_swap again
        await _axi_write(ctx, dut, REG_CTRL, 64)
        await ctx.tick()

        s2 = await _axi_read(ctx, dut, REG_STATUS)
        bank_reads.append((s2 >> 2) & 0x1)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert bank_reads == [0, 1, 0], f"Expected [0, 1, 0], got {bank_reads}"


# ==============================================================================
# 4. PCIe Driver Prefetch & Double-Buffering Integration Test
# ==============================================================================
def test_pcie_driver_prefetch_and_double_buffering():
    """Verify MockPCIeBridge and RailNetPCIeDriver prefetch_layer pipeline."""
    driver = RailNetPCIeDriver("railnet_test_db")
    driver.enable_double_buffering()

    c0 = _make_dummy_compiled(in_features=16, out_features=8, seed=10)
    c1 = _make_dummy_compiled(in_features=16, out_features=8, seed=20)

    x = np.random.randn(16).astype(np.float64)

    # Program Layer 0 into active bank (Bank 0)
    driver.bridge.program_tensor(c0, bank=0)
    # Preload Layer 1 into Bank 1 and arm the swap
    driver.prefetch_layer(c1)

    # Check status: bank_ready bit 3 should be 1
    status = driver.read_csr(REG_STATUS)
    assert (status >> 3) & 0x1 == 1, "bank_ready not set after prefetch_layer"

    # Execute Layer 0
    y0_hw = driver.dispatch_linear(x, c0)
    y0_ref = rail_linear_fast(x, c0)
    np.testing.assert_allclose(y0_hw, y0_ref, rtol=1e-5, atol=1e-5)

    # After execution, autonomous swap should have toggled active_bank to 1
    status = driver.read_csr(REG_STATUS)
    active_bank = (status >> 2) & 0x1
    assert active_bank == 1, "active_bank did not swap to 1"

    # Execute Layer 1 on swapped bank
    y1_hw = driver.dispatch_linear(x, c1)
    y1_ref = rail_linear_fast(x, c1)
    np.testing.assert_allclose(y1_hw, y1_ref, rtol=1e-5, atol=1e-5)

    driver.close()


# ==============================================================================
# 5. Background Prefetch Worker Thread Test
# ==============================================================================
def test_driver_background_prefetch_queue():
    """Verify asynchronous prefetch queue with background worker thread."""
    driver = RailNetPCIeDriver("railnet_test_thread")
    driver.enable_double_buffering()

    c0 = _make_dummy_compiled(in_features=16, out_features=8, seed=30)
    c1 = _make_dummy_compiled(in_features=16, out_features=8, seed=40)
    c2 = _make_dummy_compiled(in_features=16, out_features=8, seed=50)

    x = np.random.randn(16).astype(np.float64)

    # Load initial layer
    driver.bridge.program_tensor(c0, bank=0)

    # Queue next layer into background worker thread
    driver.queue_prefetch(c1)
    time.sleep(0.1)  # Allow background thread to process and arm next bank

    # Check that bank_ready is armed
    status = driver.read_csr(REG_STATUS)
    assert (status >> 3) & 0x1 == 1, "Worker thread failed to arm next bank"

    # Execute Layer 0 -> triggers autonomous bank swap
    y0_hw = driver.dispatch_linear(x, c0)
    np.testing.assert_allclose(y0_hw, rail_linear_fast(x, c0), rtol=1e-5, atol=1e-5)

    # Queue Layer 2
    driver.queue_prefetch(c2)
    time.sleep(0.1)

    # Execute Layer 1 -> triggers autonomous bank swap to Layer 2
    y1_hw = driver.dispatch_linear(x, c1)
    np.testing.assert_allclose(y1_hw, rail_linear_fast(x, c1), rtol=1e-5, atol=1e-5)

    driver.close()
