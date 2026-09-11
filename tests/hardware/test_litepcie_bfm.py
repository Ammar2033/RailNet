"""Gate 2 LitePCIe Host BFM Simulation Suite (Checklist 1.1).

Simulates LitePCIe open-source PCIe endpoint without physical hardware:
- LitePCIeHostBFM: Behavioral model of LitePCIe BAR0 + DMA writer/reader queues
- Tests DMA descriptor chunking (4096B), CSR retry/backoff, timeout+soft-reset recovery
- Tests in-band program_layer (rails/codebook/routes) via CSR (REG_PROG_*)
- Validates hardened LitePCIeBridge / RailNetPCIeDriver logic in pure simulation

All tests run on Mock/Fake FDs - no /dev/litepcie* required, evidence tier SIMULATED.
Physical hardware will reuse same CSR offsets and chunking contract.
"""

import os
import struct
import queue
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from railnet.kernel import CompiledTensor, prepare, rail_linear_fast
from railnet.runtime.pcie import (
    LitePCIeBridge,
    MockPCIeBridge,
    RailNetPCIeDriver,
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


class LitePCIeHostBFM:
    """Behavioral LitePCIe Host Bus Functional Model (BFM).

    Emulates LitePCIe BAR0 CSR (mmap) and DMA writer/reader char devices
    as queues in memory. Mirrors LitePCIeBridge hardware contract:
    - CSR: 32-bit AXI4-Lite, offsets 0x00-0x20
    - DMA H2C: Host writes int16 activations via writer queue, BFM computes via MockPCIeBridge
    - DMA C2H: BFM pushes int32 results to reader queue
    - Link training: REG_STATUS[15:8] == num_tiles
    """

    def __init__(self, num_tiles: int = 4, dma_chunk_bytes: int = 4096, inject_csr_faults: int = 0):
        self.num_tiles = num_tiles
        self.dma_chunk_bytes = dma_chunk_bytes
        self.inject_csr_faults = inject_csr_faults
        self._csr_fault_counter = 0
        self.mock = MockPCIeBridge(num_tiles=num_tiles)
        # DMA queues emulate char device FIFOs
        self.dma_h2c_queue: queue.Queue = queue.Queue()
        self.dma_c2h_queue: queue.Queue = queue.Queue()
        self.dma_stats = {"bytes_h2c": 0, "bytes_c2h": 0, "transfers": 0, "retries": 0}

    def write_csr(self, offset: int, value: int) -> None:
        # Inject transient fault for retry test
        if self.inject_csr_faults > 0 and self._csr_fault_counter < self.inject_csr_faults:
            self._csr_fault_counter += 1
            raise BlockingIOError(f"Injected CSR fault {self._csr_fault_counter}/{self.inject_csr_faults}")
        self.mock.write_csr(offset, value)

    def read_csr(self, offset: int) -> int:
        if self.inject_csr_faults > 0 and self._csr_fault_counter < self.inject_csr_faults:
            self._csr_fault_counter += 1
            raise BlockingIOError(f"Injected CSR fault {self._csr_fault_counter}/{self.inject_csr_faults}")
        return self.mock.read_csr(offset)

    def dma_write_h2c(self, payload: bytes) -> int:
        """Host DMA writer: chunk payload and enqueue."""
        total = 0
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + self.dma_chunk_bytes]
            self.dma_h2c_queue.put(chunk)
            self.dma_stats["bytes_h2c"] += len(chunk)
            total += len(chunk)
            offset += len(chunk)
        self.dma_stats["transfers"] += 1
        # BFM immediately processes: dequeue H2C, compute, enqueue C2H
        self._process_dma()
        return total

    def dma_read_c2h(self, num_bytes: int, timeout: float = 1.0) -> bytes:
        t0 = time.monotonic()
        buf = bytearray()
        while len(buf) < num_bytes:
            try:
                chunk = self.dma_c2h_queue.get(timeout=0.05)
                buf.extend(chunk)
                self.dma_stats["bytes_c2h"] += len(chunk)
            except queue.Empty:
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"BFM C2H timeout {len(buf)}/{num_bytes}B")
                continue
        return bytes(buf)

    def _process_dma(self) -> None:
        """BFM compute: drain H2C queue, reconstruct activations, run Mock compute, push results."""
        # Drain all H2C chunks into single activation buffer
        h2c_data = bytearray()
        while not self.dma_h2c_queue.empty():
            try:
                h2c_data.extend(self.dma_h2c_queue.get_nowait())
            except queue.Empty:
                break
        if not h2c_data:
            return
        # Convert bytes -> int16 activations (BFM assumes int16 input)
        x = np.frombuffer(bytes(h2c_data), dtype=np.int16).astype(np.float64)
        # Let Mock handle compute (it uses compiled tensor if programmed)
        self.mock._last_activations = x
        self.mock.regs[REG_IN_FEATURES] = len(x)
        # Trigger compute via dma_transfer (updates cycle counter, status, bank swap)
        results = self.mock.dma_transfer(x)
        # Push results as int32 bytes via C2H
        result_bytes = results.astype(np.int32).tobytes()
        # Chunk C2H as well
        offset = 0
        while offset < len(result_bytes):
            chunk = result_bytes[offset : offset + self.dma_chunk_bytes]
            self.dma_c2h_queue.put(chunk)
            offset += len(chunk)

    def get_link_status(self) -> dict:
        status = self.mock.read_csr(REG_STATUS)
        return {
            "is_connected": True,
            "link_up": ((status >> 8) & 0xFF) == self.num_tiles,
            "num_tiles": (status >> 8) & 0xFF,
            "status_reg": status,
        }


def _make_dummy_compiled(out_features=4, in_features=32, rails=8, seed=42):
    rng = np.random.RandomState(seed)
    c = CompiledTensor.__new__(CompiledTensor)
    c.checksum_ok = True
    c.tensor_name = "bfm_dummy"
    c.rail_count = rails
    c.max_terms = 2
    c.shape = (out_features, in_features)
    c.out_features = out_features
    c.in_features = in_features
    c.load_seconds = 0.0
    c.scale = 1.0
    # Use integer-valued rails so that DMA int32 truncation is lossless (BFM models 32-bit signed m_axis_tdata)
    # Random ints in [-5,5] give deterministic integer outputs, avoiding float truncation mismatch
    c.rails_f64 = rng.randint(-5, 6, size=rails).astype(np.float64)
    # Ensure no zero rails for better coverage (avoid degenerate)
    c.rails_f64[c.rails_f64 == 0] = 1
    c.term_rail = np.zeros((65536, 2), dtype=np.int32)
    c.term_sign = np.zeros((65536, 2), dtype=np.int8)
    c.term_active = np.zeros((65536, 2), dtype=bool)
    for g in range(32):
        r = rng.choice(rails, size=1)
        c.term_rail[g, 0] = r[0]
        c.term_sign[g, 0] = 1
        c.term_active[g, 0] = True
    c.route_ids = rng.randint(0, 32, size=(out_features, in_features)).astype(np.int32)
    c.prepared = False
    prepare(c)
    # Add rails_int32 for program_layer (consistent with int rails)
    c.rails_int32 = c.rails_f64.astype(np.int32)
    return c


# ---------------------------------------------------------------------------
# Test 1: BFM CSR Basic + Link Probe
# ---------------------------------------------------------------------------
def test_litepcie_bfm_csr_basic_and_link_probe():
    bfm = LitePCIeHostBFM(num_tiles=4)
    # Link probe: status[15:8] == num_tiles
    status = bfm.read_csr(REG_STATUS)
    assert (status >> 8) == 4
    assert bfm.get_link_status()["link_up"] is True

    # CSR retry: write/read with no faults
    bfm.write_csr(REG_TILE_MASK, 0x0A)
    assert bfm.read_csr(REG_TILE_MASK) == 0x0A
    bfm.write_csr(REG_IN_FEATURES, 128)
    assert bfm.read_csr(REG_IN_FEATURES) == 128

    # LitePCIeBridge hardened retry simulation: inject 2 transient faults then success
    bfm_faulty = LitePCIeHostBFM(num_tiles=4, inject_csr_faults=2)
    # First 2 writes should raise, 3rd succeed - simulate driver retry loop
    retries = 0
    for attempt in range(3):
        try:
            bfm_faulty.write_csr(REG_TILE_MASK, 0x0F)
            break
        except BlockingIOError:
            retries += 1
            time.sleep(0.001 * (2 ** attempt))
    assert retries == 2
    assert bfm_faulty.read_csr(REG_TILE_MASK) == 0x0F


# ---------------------------------------------------------------------------
# Test 2: DMA Chunking (4096B) + Bit-Exact
# ---------------------------------------------------------------------------
def test_litepcie_bfm_dma_chunking_bit_exact():
    bfm = LitePCIeHostBFM(num_tiles=4, dma_chunk_bytes=4096)
    c = _make_dummy_compiled(out_features=4, in_features=32)
    # Program BFM mock with compiled tensor (simulates CSR in-band programming)
    bfm.mock.program_tensor(c)

    # Small payload: K=32 (< chunk) -> 64 bytes, single chunk
    rng = np.random.default_rng(123)
    x_small = rng.integers(-50, 50, size=32, dtype=np.int16).astype(np.float64)
    payload_small = x_small.astype(np.int16).tobytes()
    assert len(payload_small) == 64
    n = bfm.dma_write_h2c(payload_small)
    assert n == 64
    assert bfm.dma_stats["transfers"] == 1
    result_bytes = bfm.dma_read_c2h(4 * 4)
    y_bfm = np.frombuffer(result_bytes, dtype=np.int32).astype(np.float64)
    y_ref = rail_linear_fast(x_small, c)
    np.testing.assert_allclose(y_bfm, y_ref[:4])

    # Large payload: K=4096 -> 8192 bytes, requires 2 chunks with 4096B chunk size
    bfm2 = LitePCIeHostBFM(num_tiles=4, dma_chunk_bytes=4096)
    c2 = _make_dummy_compiled(out_features=4, in_features=4096)
    bfm2.mock.program_tensor(c2)
    x_large = rng.integers(-50, 50, size=4096, dtype=np.int16).astype(np.float64)
    payload_large = x_large.astype(np.int16).tobytes()
    assert len(payload_large) == 8192
    bfm2.dma_write_h2c(payload_large)
    # Should have created 2 H2C queue entries internally, but BFM coalesces
    assert bfm2.dma_stats["bytes_h2c"] == 8192
    result_bytes2 = bfm2.dma_read_c2h(4 * 4)
    y_bfm2 = np.frombuffer(result_bytes2, dtype=np.int32).astype(np.float64)
    y_ref2 = rail_linear_fast(x_large, c2)
    np.testing.assert_allclose(y_bfm2, y_ref2[:4])

    # Edge: exact chunk boundary K=2048 -> 4096 bytes == 1 chunk
    bfm3 = LitePCIeHostBFM(num_tiles=4, dma_chunk_bytes=4096)
    c3 = _make_dummy_compiled(out_features=4, in_features=2048)
    bfm3.mock.program_tensor(c3)
    x_edge = rng.integers(-50, 50, size=2048, dtype=np.int16).astype(np.float64)
    bfm3.dma_write_h2c(x_edge.astype(np.int16).tobytes())
    assert bfm3.dma_stats["bytes_h2c"] == 4096
    y_edge = np.frombuffer(bfm3.dma_read_c2h(16), dtype=np.int32).astype(np.float64)
    y_ref_edge = rail_linear_fast(x_edge, c3)
    np.testing.assert_allclose(y_edge, y_ref_edge[:4])


# ---------------------------------------------------------------------------
# Test 3: Timeout + Soft-Reset Recovery
# ---------------------------------------------------------------------------
def test_litepcie_bfm_timeout_and_soft_reset_recovery():
    bfm = LitePCIeHostBFM(num_tiles=2)
    # Start transaction then soft-reset before completion
    bfm.write_csr(REG_CTRL, 1)  # start
    status = bfm.read_csr(REG_STATUS)
    assert (status & 0x1) == 1  # busy

    # Simulate BFM hang: C2H queue empty -> read should timeout
    with pytest.raises(TimeoutError):
        bfm.dma_read_c2h(2 * 4, timeout=0.1)

    # Issue soft-reset (REG_CTRL bit1)
    bfm.write_csr(REG_CTRL, 0x2)
    status_after = bfm.read_csr(REG_STATUS)
    assert (status_after & 0x1) == 0  # busy cleared
    # After reset, new transaction should succeed
    c = _make_dummy_compiled(out_features=2, in_features=16)
    bfm.mock.program_tensor(c)
    x = np.random.default_rng(0).integers(-10, 10, size=16, dtype=np.int16).astype(np.float64)
    bfm.dma_write_h2c(x.astype(np.int16).tobytes())
    rb = bfm.dma_read_c2h(8)
    assert len(rb) == 8


# ---------------------------------------------------------------------------
# Test 4: In-Band Program Layer via CSR (Rails/Codebook/Routes)
# ---------------------------------------------------------------------------
def test_litepcie_bfm_in_band_program_layer():
    # Use real driver with BFM-backed Mock to test program_layer contract
    # We simulate LitePCIe CSR in-band by directly using Mock's CSR interface via driver
    bfm = LitePCIeHostBFM(num_tiles=4)
    # Create a driver that delegates CSR to BFM (simulate LitePCIe BAR0)
    driver = RailNetPCIeDriver("bfm_test_inband", backend="mock")
    # Replace driver's bridge with BFM's mock and wire CSR
    driver.bridge = bfm.mock
    driver.is_hardware = False  # keep mock path but test program_layer logic
    # Force program_layer to go via BFM mock's CSR (we monkeypatch write_csr to count)
    orig_write = bfm.mock.write_csr
    write_log = []

    def logging_write(offset, value):
        write_log.append((offset, value))
        return orig_write(offset, value)

    bfm.mock.write_csr = logging_write

    c = _make_dummy_compiled(out_features=4, in_features=16, rails=8)
    # Simulate what LitePCIeBridge program_layer would do: write via CSR
    # Here we call driver.program_layer which for mock just caches, so we directly test BFM CSR
    # Instead test low-level: program rails via CSR sequence as driver does in hardware mode
    driver.is_hardware = True
    driver.write_csr = bfm.mock.write_csr  # type: ignore
    driver.read_csr = bfm.mock.read_csr  # type: ignore
    driver.bridge = bfm.mock
    # Mock status to report 4 tiles
    bfm.mock.regs[REG_STATUS] = (4 << 8)
    # Now call hardware program_layer path
    # Need to ensure compiled has required fields for program_layer
    c.rails_int32 = np.arange(8, dtype=np.int32)
    c.term_rail = np.zeros((32, 2), dtype=np.int32)
    c.term_sign = np.zeros((32, 2), dtype=np.int8)
    c.term_active = np.zeros((32, 2), dtype=bool)
    for i in range(8):
        c.term_rail[i, 0] = i
        c.term_sign[i, 0] = 1
        c.term_active[i, 0] = True
    c.route_ids = np.arange(16, dtype=np.int32).reshape(1, -1).repeat(4, axis=0)
    # Call program_layer (hardware path)
    driver.program_layer(c)

    # Verify BFM received rails, codebook, routes via CSR in-band
    # Check rails: at least one per tile
    for t in range(4):
        assert (t, 0) in bfm.mock.bram_rails[0] or (t, 0) in bfm.mock.bram_rails[1] or any(k[0] == t for k in bfm.mock.bram_rails[0].keys())
    # Routes: check (t, 0) exists
    for t in range(4):
        # Routes are stored per tile, addr 0..15
        assert (t, 0) in bfm.mock.bram_routes[0] or (t, 0) in bfm.mock.bram_routes[1] or any(k[0] == t for k in bfm.mock.bram_routes[0].keys())
    # At least one codebook entry
    assert len(bfm.mock.bram_cb[0]) > 0 or len(bfm.mock.bram_cb[1]) > 0
    # Verify CSR writes included prog strobe (REG_PROG_CTRL)
    prog_strobes = [v for (off, v) in write_log if off == REG_PROG_CTRL and v == 1]
    assert len(prog_strobes) >= 4 * (8 + 8 + 16)  # rails + cb + routes approx


# ---------------------------------------------------------------------------
# Test 5: Hardened LitePCIeBridge fallback + Driver link status (no hardware)
# ---------------------------------------------------------------------------
def test_hardened_bridge_fallback_and_link_status():
    # LitePCIeBridge with nonexistent path should be not connected but not raise
    bridge = LitePCIeBridge("/dev/nonexistent_litetest_xyz", verbose=False)
    assert bridge.is_connected is False
    assert bridge.link_up is False
    status = bridge.get_link_status()
    assert status["is_connected"] is False
    assert status["backend"] == "litepcie"
    # Driver auto should fallback to mock
    drv = RailNetPCIeDriver("nonexistent_bfm_auto_xyz", backend="auto", verbose=False)
    assert drv.hardware_backend == "mock"
    assert drv.is_hardware is False
    ls = drv.get_link_status()
    assert ls["backend"] == "mock"
    assert ls["link_up"] is True  # mock always up
    assert "dma_stats" in ls
    assert drv.get_dma_stats()["ops"] == 0
    assert drv.run_self_test() is True
    drv.close()

    # Explicit litepcie backend with missing device should raise
    with pytest.raises(RuntimeError, match="LitePCIe hardware device node not found"):
        RailNetPCIeDriver("nonexistent_bfm_xyz", backend="litepcie")

    # FTDI fallback should not be hardware unless env flag
    drv2 = RailNetPCIeDriver("nonexistent_ftdi_auto", backend="auto", verbose=False)
    # Should still be mock, not ftdi, because no /dev/railnet_ftdi and no env
    assert drv2.hardware_backend == "mock"
    drv2.close()
