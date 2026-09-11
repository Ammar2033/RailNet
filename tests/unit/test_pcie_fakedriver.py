"""2.1 Fake-FD Unit Tests for Hardened PCIe Driver (Checklist 2.1).

Tests the hardened retry/backoff, link-status and self-test paths of
railnet/runtime/pcie.py without physical hardware, using direct
injection of fake mmap/FD objects.

Covers:
- LitePCIe BAR0 mmap retry (BlockingIOError -> retry -> success)
- XDMA CSR retry via os.lseek/os.write
- get_link_status() bridge_status / xdma_status_reg
- run_self_test() loopback via REG_TILE_MASK

Evidence tier: SIMULATED (injected FD), no hardware required.
"""

import os
import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from railnet.runtime.pcie import (
    LitePCIeBridge,
    MockPCIeBridge,
    RailNetPCIeDriver,
    REG_TILE_MASK,
    REG_STATUS,
    REG_CTRL,
)


class FakeMmap:
    """Minimal mmap fake for LitePCIe BAR0 (4096B). Stores regs as dict."""

    def __init__(self, size=4096):
        self.size = size
        self.regs = {REG_STATUS: (4 << 8)}
        self._pos = 0
        self.closed = False

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset

    def write(self, data: bytes):
        if len(data) == 4:
            val = struct.unpack("<I", data)[0]
            self.regs[self._pos] = val
        self._pos += len(data)

    def read(self, n: int) -> bytes:
        val = self.regs.get(self._pos, 0)
        if self._pos == REG_STATUS and val == 0:
            val = (4 << 8)
        self._pos += n
        return struct.pack("<I", val & 0xFFFFFFFF)[:n]

    def flush(self):
        pass

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Test 1: LitePCIe retry (first 2 writes BlockingIOError, 3rd success)
# ---------------------------------------------------------------------------
def test_driver_litepcie_retry():
    # Directly inject fake mmap into a LitePCIeBridge without going through
    # hardware discovery (which is OS-dependent and flaky on Windows)
    bridge = LitePCIeBridge.__new__(LitePCIeBridge)
    # Manually set up minimal state as if __init__ had succeeded
    bridge.dev_path = "/dev/litepcie0"
    bridge.ctrl_path = "/dev/litepcie0"
    bridge.dma_writer_path = None
    bridge.dma_reader_path = None
    bridge.bar_size = 0x1000
    bridge.dma_chunk_bytes = 4096
    bridge.verbose = False
    bridge._fd_ctrl = 999
    bridge._mmap = FakeMmap()
    bridge._mmap.regs[REG_STATUS] = (4 << 8)
    bridge.is_connected = True
    bridge.is_dma_capable = False
    bridge.link_up = True
    bridge.dma_stats = {"bytes_h2c": 0, "bytes_c2h": 0, "transfers": 0, "errors": 0, "retries": 0}
    bridge._last_status = (4 << 8)
    import threading

    bridge._lock = threading.RLock()

    # Inject transient fault: first 2 mmap.write should fail
    call_count = {"n": 0}
    orig_write = bridge._mmap.write

    def faulty_write(data):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise BlockingIOError("injected fault")
        return orig_write(data)

    bridge._mmap.write = faulty_write  # type: ignore

    # Now write via bridge - should retry and succeed on 3rd attempt
    bridge.write_csr(REG_TILE_MASK, 0x0A)
    assert bridge.dma_stats["retries"] >= 2
    assert bridge._mmap.regs[REG_TILE_MASK] == 0x0A

    # Read also with retry: first 2 reads fail
    call_count["n"] = 0
    orig_read = bridge._mmap.read

    def faulty_read(n):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise BlockingIOError("injected read fault")
        return orig_read(n)

    bridge._mmap.read = faulty_read  # type: ignore
    val = bridge.read_csr(REG_TILE_MASK)
    assert val == 0x0A
    assert bridge.dma_stats["retries"] >= 4

    bridge.close()


def test_driver_xdma_retry(monkeypatch):
    # Simulate XDMA device by directly injecting fake FDs into driver
    # Avoid Path.exists mocking on Windows; directly construct driver with mock then patch to XDMA
    drv = RailNetPCIeDriver("railnet0_xdma_retry_test", backend="mock", verbose=False)
    # Convert to XDMA mode manually
    fake_fd_user = 1001
    fake_fd_h2c = 1002
    fake_fd_c2h = 1003
    drv._fd_user = fake_fd_user
    drv._fd_h2c = fake_fd_h2c
    drv._fd_c2h = fake_fd_c2h
    drv.hardware_backend = "xdma"
    drv.is_hardware = True
    drv.link_up = True

    fd_to_regs = {fake_fd_user: {REG_STATUS: (4 << 8)}}
    lseek_pos = {}

    write_attempts = {"n": 0}

    def fake_lseek(fd, pos, whence):
        lseek_pos[fd] = pos
        return pos

    def fake_write(fd, data):
        write_attempts["n"] += 1
        if write_attempts["n"] <= 2:
            raise BlockingIOError("injected XDMA write fault")
        pos = lseek_pos.get(fd, 0)
        if fd == fake_fd_user and len(data) == 4:
            val = struct.unpack("<I", data)[0]
            fd_to_regs[fd][pos] = val
        return len(data)

    def fake_read(fd, n):
        pos = lseek_pos.get(fd, 0)
        val = fd_to_regs[fd].get(pos, (4 << 8) if pos == REG_STATUS else 0)
        return struct.pack("<I", val)[:n]

    monkeypatch.setattr(os, "lseek", fake_lseek)
    monkeypatch.setattr(os, "write", fake_write)
    monkeypatch.setattr(os, "read", fake_read)
    monkeypatch.setattr(os, "close", lambda fd: None)

    # write_csr should have retried 2 times and succeeded
    drv.write_csr(REG_TILE_MASK, 0x0B)
    assert write_attempts["n"] >= 3
    assert drv.dma_stats["retries"] >= 2
    assert fd_to_regs[fake_fd_user][REG_TILE_MASK] == 0x0B

    val = drv.read_csr(REG_TILE_MASK)
    assert val == 0x0B

    drv.close()


# ---------------------------------------------------------------------------
# Test 2: get_link_status
# ---------------------------------------------------------------------------
def test_driver_get_link_status():
    # Use mock fallback - get_link_status should still return bridge_status
    drv = RailNetPCIeDriver("nonexistent_link_status_xyz", backend="mock", verbose=False)
    ls = drv.get_link_status()
    assert ls["backend"] == "mock"
    assert ls["link_up"] is True
    assert "bridge_status" in ls
    assert ls["bridge_status"]["backend"] == "mock"
    assert "dma_stats" in ls
    drv.close()

    # LitePCIe fake hardware via direct injection
    bridge = LitePCIeBridge.__new__(LitePCIeBridge)
    bridge.dev_path = "/dev/litepcie0"
    bridge.ctrl_path = "/dev/litepcie0"
    bridge.bar_size = 0x1000
    bridge.dma_chunk_bytes = 4096
    bridge.verbose = False
    bridge._fd_ctrl = 999
    bridge._mmap = FakeMmap()
    bridge._mmap.regs[REG_STATUS] = (4 << 8)
    bridge.is_connected = True
    bridge.is_dma_capable = True
    bridge.link_up = True
    bridge.dma_stats = {"bytes_h2c": 0, "bytes_c2h": 0, "transfers": 0, "errors": 0, "retries": 0}
    bridge._last_status = (4 << 8)
    import threading

    bridge._lock = threading.RLock()
    # Also need dma paths for status
    bridge.dma_writer_path = "/dev/litepcie0_dma_writer"
    bridge.dma_reader_path = "/dev/litepcie0_dma_reader"

    status = bridge.get_link_status()
    assert status["is_connected"] is True
    assert status["backend"] == "litepcie"
    assert status["num_tiles"] == 4
    bridge.close()


def test_driver_get_dma_stats():
    drv = RailNetPCIeDriver("nonexistent_dma_stats_xyz", backend="mock", verbose=False)
    stats = drv.get_dma_stats()
    assert stats["ops"] == 0
    assert "bridge" in stats
    drv.write_csr(REG_TILE_MASK, 0x01)
    drv.read_csr(REG_TILE_MASK)
    stats2 = drv.get_dma_stats()
    assert "bridge" in stats2
    drv.close()


# ---------------------------------------------------------------------------
# Test 3: run_self_test
# ---------------------------------------------------------------------------
def test_driver_run_self_test():
    drv = RailNetPCIeDriver("nonexistent_selftest_xyz", backend="mock", verbose=False)
    assert drv.run_self_test(0xA5) is True
    assert drv.run_self_test(0x55) is True
    drv.write_csr(REG_TILE_MASK, 0x0F)
    assert drv.read_csr(REG_TILE_MASK) == 0x0F
    drv.close()

    # LitePCIe fake hardware self-test via direct bridge
    fake_mmap2 = FakeMmap()
    fake_mmap2.regs[REG_STATUS] = (4 << 8)
    fake_mmap2.regs[REG_TILE_MASK] = 0x00

    bridge = LitePCIeBridge.__new__(LitePCIeBridge)
    bridge.dev_path = "/dev/litepcie0"
    bridge.ctrl_path = "/dev/litepcie0"
    bridge.bar_size = 0x1000
    bridge.dma_chunk_bytes = 4096
    bridge.verbose = False
    bridge._fd_ctrl = 999
    bridge._mmap = fake_mmap2
    bridge.is_connected = True
    bridge.is_dma_capable = False
    bridge.link_up = True
    bridge.dma_stats = {"bytes_h2c": 0, "bytes_c2h": 0, "transfers": 0, "errors": 0, "retries": 0}
    bridge._last_status = (4 << 8)
    import threading

    bridge._lock = threading.RLock()
    bridge.dma_writer_path = None
    bridge.dma_reader_path = None

    assert bridge.dma_self_test(0xA5) is True
    assert bridge.read_csr(REG_TILE_MASK) == 0x00
    bridge.close()
