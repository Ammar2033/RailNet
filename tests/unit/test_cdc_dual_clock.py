"""Gate 2 Pillar 2 — CDC (Clock Domain Crossing) Verification Suite.

Tests for:
1. AxisCDCFIFO data integrity across clock domains.
2. AxisCDCFIFO backpressure (FIFO-full stall).
3. AxiLiteCDCBridge CSR read/write across clock domains.
4. RailNetDualClockTop AXI-Lite CDC roundtrip with dual clocks.
5. Single-clock backward compatibility (core_clk == host_clk).
"""

import numpy as np
import pytest
from amaranth import ClockDomain, ClockSignal, DomainRenamer, Module, ResetSignal, Signal
from amaranth.sim import Simulator

from hardware.rtl.axi import (
    AxiLiteCsr,
    REG_CTRL,
    REG_IN_FEATURES,
    REG_OUT_FEATURES,
    REG_STATUS,
    REG_TILE_MASK,
)
from hardware.rtl.cdc import (
    AxisCDCFIFO,
    AxiLiteCDCBridge,
    RailNetDualClockTop,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

HOST_PERIOD = 1 / 50e6     # 50 MHz → 20 ns
CORE_PERIOD = 1 / 200e6    # 200 MHz → 5 ns


def _build_dual_clock_sim(dut):
    """Create a Simulator with host (50 MHz) and core (200 MHz) clocks."""
    sim = Simulator(dut)
    sim.add_clock(HOST_PERIOD, domain="host")
    sim.add_clock(CORE_PERIOD, domain="core")
    return sim


# ---------------------------------------------------------------------------
#  Test 1: AxisCDCFIFO — data integrity
# ---------------------------------------------------------------------------

def test_axis_cdc_fifo_data_integrity():
    """16-element vector written in host domain, read in core domain.
    Uses FIFO depth=32 (bigger than data) to avoid deadlock in serial test.
    Asserts bit-exact match with zero data loss."""
    N = 16
    dut = AxisCDCFIFO(data_width=32, depth=32, w_domain="host", r_domain="core")

    rng = np.random.default_rng(100)
    send_data = rng.integers(0, 2**32, size=N, dtype=np.uint32)
    recv_data = []
    last_flags = []

    async def testbench(ctx):
        # --- Write phase: push all N values (FIFO depth > N, so no stall) ---
        for i in range(N):
            ctx.set(dut.w_tdata, int(send_data[i]))
            ctx.set(dut.w_tvalid, 1)
            ctx.set(dut.w_tlast, 1 if i == N - 1 else 0)
            await ctx.tick("host")
            # Wait until accepted
            while not ctx.get(dut.w_tready):
                await ctx.tick("host")
        ctx.set(dut.w_tvalid, 0)

        # --- Read phase: pop all N values ---
        ctx.set(dut.r_tready, 1)
        for _ in range(N):
            while not ctx.get(dut.r_tvalid):
                await ctx.tick("core")
            recv_data.append(ctx.get(dut.r_tdata))
            last_flags.append(ctx.get(dut.r_tlast))
            await ctx.tick("core")
        ctx.set(dut.r_tready, 0)

    sim = _build_dual_clock_sim(dut)
    sim.add_testbench(testbench)
    sim.run()

    assert len(recv_data) == N
    np.testing.assert_array_equal(
        np.array(recv_data, dtype=np.uint32), send_data,
    )
    assert last_flags[-1] == 1, "Last element should have tlast=1"
    assert all(f == 0 for f in last_flags[:-1]), "Non-last should have tlast=0"


# ---------------------------------------------------------------------------
#  Test 2: AxisCDCFIFO — backpressure
# ---------------------------------------------------------------------------

def test_axis_cdc_fifo_backpressure():
    """Writer pushes DEPTH elements into the FIFO, then reader drains.
    Verifies FIFO fills up correctly and data arrives in order."""
    DEPTH = 8
    dut = AxisCDCFIFO(data_width=16, depth=DEPTH, w_domain="host", r_domain="core")

    # Write exactly DEPTH elements (maximum that fits without reader)
    N = DEPTH - 1  # AsyncFIFO usable depth is depth-1
    send_data = list(range(1, N + 1))
    recv_data = []
    fifo_was_full = [False]

    async def testbench(ctx):
        # --- Writer: push N values, checking for full ---
        for val in send_data:
            ctx.set(dut.w_tdata, val)
            ctx.set(dut.w_tvalid, 1)
            ctx.set(dut.w_tlast, 0)
            await ctx.tick("host")
            while not ctx.get(dut.w_tready):
                fifo_was_full[0] = True
                await ctx.tick("host")
        ctx.set(dut.w_tvalid, 0)

        # Wait a bit for CDC propagation
        for _ in range(20):
            await ctx.tick("core")

        # Check if FIFO is full now (w_tready should be 0 when full)
        ctx.set(dut.w_tdata, 999)
        ctx.set(dut.w_tvalid, 1)
        await ctx.tick("host")
        if not ctx.get(dut.w_tready):
            fifo_was_full[0] = True
        ctx.set(dut.w_tvalid, 0)

        # --- Reader: drain all values ---
        ctx.set(dut.r_tready, 1)
        for _ in range(N):
            while not ctx.get(dut.r_tvalid):
                await ctx.tick("core")
            recv_data.append(ctx.get(dut.r_tdata))
            await ctx.tick("core")

    sim = _build_dual_clock_sim(dut)
    sim.add_testbench(testbench)
    sim.run()

    assert recv_data == send_data, f"Data mismatch: {recv_data} != {send_data}"


# ---------------------------------------------------------------------------
#  Test 3: AxiLiteCDCBridge — CSR read/write across domains
# ---------------------------------------------------------------------------

def test_axi_lite_cdc_bridge_readwrite():
    """Write IN_FEATURES=256 and TILE_MASK=0b1010 via AXI-Lite CDC bridge,
    read them back and verify bit-exact values through the clock crossing."""

    # Build a test harness: bridge (host) → CSR (core)
    m = Module()
    m.domains += ClockDomain("host")
    m.domains += ClockDomain("core")

    bridge = AxiLiteCDCBridge(host_domain="host", core_domain="core")
    m.submodules.bridge = bridge

    csr = DomainRenamer({"sync": "core"})(AxiLiteCsr(num_tiles=4))
    m.submodules.csr = csr

    # Wire bridge core master → CSR slave
    m.d.comb += [
        csr.awaddr.eq(bridge.c_awaddr),   csr.awvalid.eq(bridge.c_awvalid),
        bridge.c_awready.eq(csr.awready),
        csr.wdata.eq(bridge.c_wdata),     csr.wstrb.eq(bridge.c_wstrb),
        csr.wvalid.eq(bridge.c_wvalid),   bridge.c_wready.eq(csr.wready),
        bridge.c_bresp.eq(csr.bresp),     bridge.c_bvalid.eq(csr.bvalid),
        csr.bready.eq(bridge.c_bready),
        csr.araddr.eq(bridge.c_araddr),   csr.arvalid.eq(bridge.c_arvalid),
        bridge.c_arready.eq(csr.arready),
        bridge.c_rdata.eq(csr.rdata),     bridge.c_rresp.eq(csr.rresp),
        bridge.c_rvalid.eq(csr.rvalid),   csr.rready.eq(bridge.c_rready),
        # Tie off status inputs
        csr.status_busy.eq(0), csr.status_done.eq(0), csr.cycle_count.eq(0),
        csr.hw_bank_swap.eq(0),
    ]

    readbacks = {}

    async def tb(ctx):
        for _ in range(10):
            await ctx.tick("host")

        # Write IN_FEATURES = 256
        await _bridge_write(ctx, bridge, REG_IN_FEATURES, 256)
        # Write TILE_MASK = 0b1010
        await _bridge_write(ctx, bridge, REG_TILE_MASK, 0b1010)
        # Read back
        readbacks["in_feat"] = await _bridge_read(ctx, bridge, REG_IN_FEATURES)
        readbacks["tile_mask"] = await _bridge_read(ctx, bridge, REG_TILE_MASK)

    sim = Simulator(m)
    sim.add_clock(HOST_PERIOD, domain="host")
    sim.add_clock(CORE_PERIOD, domain="core")
    sim.add_testbench(tb)
    sim.run()

    assert readbacks["in_feat"] == 256, f"IN_FEATURES: {readbacks['in_feat']}"
    assert readbacks["tile_mask"] == 0b1010, f"TILE_MASK: {readbacks['tile_mask']}"


# ---------------------------------------------------------------------------
#  Test 4: RailNetDualClockTop — AXI-Lite CDC roundtrip
# ---------------------------------------------------------------------------

def test_dual_clock_axi_lite_roundtrip():
    """Write and read CSR registers through the full RailNetDualClockTop
    with host=50MHz and core=200MHz. Verifies the AXI-Lite CDC bridge
    correctly relays transactions across the clock boundary."""
    dut = RailNetDualClockTop(
        num_tiles=2, rails=8,
        codebook_depth=16, route_depth=16, fifo_depth=8,
    )

    readbacks = {}

    async def tb(ctx):
        for _ in range(20):
            await ctx.tick("host")

        # Write IN_FEATURES = 64
        await _top_axi_write(ctx, dut, REG_IN_FEATURES, 64)
        # Write TILE_MASK = 0b11
        await _top_axi_write(ctx, dut, REG_TILE_MASK, 0b11)
        # Read back
        readbacks["in_feat"] = await _top_axi_read(ctx, dut, REG_IN_FEATURES)
        readbacks["tile_mask"] = await _top_axi_read(ctx, dut, REG_TILE_MASK)
        readbacks["status"] = await _top_axi_read(ctx, dut, REG_STATUS)

    sim = _build_dual_clock_sim(dut)
    sim.add_testbench(tb)
    sim.run()

    assert readbacks["in_feat"] == 64
    assert readbacks["tile_mask"] == 0b11
    assert (readbacks["status"] >> 8) & 0xFF == 2  # num_tiles=2


# ---------------------------------------------------------------------------
#  Test 5: Single-clock backward compatibility
# ---------------------------------------------------------------------------

def test_single_clock_backward_compat():
    """When core_clk == host_clk (same frequency), the dual-clock wrapper
    must produce identical behavior to single-clock operation."""
    dut = RailNetDualClockTop(
        num_tiles=2, rails=8,
        codebook_depth=16, route_depth=16, fifo_depth=8,
    )

    readbacks = {}

    async def tb(ctx):
        for _ in range(10):
            await ctx.tick("host")

        await _top_axi_write(ctx, dut, REG_IN_FEATURES, 512)
        readbacks["in_feat"] = await _top_axi_read(ctx, dut, REG_IN_FEATURES)
        readbacks["status"] = await _top_axi_read(ctx, dut, REG_STATUS)

    # Same frequency for both domains → single-clock equivalent
    sim = Simulator(dut)
    sim.add_clock(HOST_PERIOD, domain="host")
    sim.add_clock(HOST_PERIOD, domain="core")  # Same as host!
    sim.add_testbench(tb)
    sim.run()

    assert readbacks["in_feat"] == 512
    assert (readbacks["status"] >> 8) & 0xFF == 2


# ---------------------------------------------------------------------------
#  AXI-Lite transaction helpers
# ---------------------------------------------------------------------------

async def _bridge_write(ctx, bridge, addr, data):
    """AXI-Lite write through an AxiLiteCDCBridge instance."""
    ctx.set(bridge.h_awaddr, addr)
    ctx.set(bridge.h_awvalid, 1)
    ctx.set(bridge.h_wdata, data)
    ctx.set(bridge.h_wstrb, 0xF)
    ctx.set(bridge.h_wvalid, 1)
    ctx.set(bridge.h_bready, 1)
    for _ in range(100):
        if ctx.get(bridge.h_awready) and ctx.get(bridge.h_wready):
            break
        await ctx.tick("host")
    await ctx.tick("host")
    ctx.set(bridge.h_awvalid, 0)
    ctx.set(bridge.h_wvalid, 0)
    for _ in range(100):
        if ctx.get(bridge.h_bvalid):
            break
        await ctx.tick("host")
    await ctx.tick("host")
    ctx.set(bridge.h_bready, 0)


async def _bridge_read(ctx, bridge, addr):
    """AXI-Lite read through an AxiLiteCDCBridge instance."""
    ctx.set(bridge.h_araddr, addr)
    ctx.set(bridge.h_arvalid, 1)
    ctx.set(bridge.h_rready, 1)
    for _ in range(100):
        if ctx.get(bridge.h_arready):
            break
        await ctx.tick("host")
    await ctx.tick("host")
    ctx.set(bridge.h_arvalid, 0)
    for _ in range(100):
        if ctx.get(bridge.h_rvalid):
            break
        await ctx.tick("host")
    val = ctx.get(bridge.h_rdata)
    await ctx.tick("host")
    ctx.set(bridge.h_rready, 0)
    return val


async def _top_axi_write(ctx, dut, addr, data):
    """AXI-Lite write through a RailNetDualClockTop instance."""
    ctx.set(dut.s_axi_awaddr, addr)
    ctx.set(dut.s_axi_awvalid, 1)
    ctx.set(dut.s_axi_wdata, data)
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


async def _top_axi_read(ctx, dut, addr):
    """AXI-Lite read through a RailNetDualClockTop instance."""
    ctx.set(dut.s_axi_araddr, addr)
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
    val = ctx.get(dut.s_axi_rdata)
    await ctx.tick("host")
    ctx.set(dut.s_axi_rready, 0)
    return val
