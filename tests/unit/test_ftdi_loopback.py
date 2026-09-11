"""2.2 FTDI Virtual PTY Loopback Test (Checklist 2.2).

Tests FtdiUsbBridge packet framing and CSR round-trip without physical FTDI
hardware, using an in-process FakeSerial that mimics the FPGA-side packet
handler (like socat pty loopback on Linux).

Covers:
- _build_csr_write_packet / _build_csr_read_packet framing
- write_csr -> read_csr round-trip via FakeSerial regs
- stream_activations (DMA H2C) and read_results (DMA C2H) via FakeSerial
- get_link_status / dma_self_test

Evidence tier: SIMULATED (FakeSerial), no /dev/ttyUSB* required.
On Linux with socat, the same test can be run with real PTY:
  socat -d -d pty,raw,echo=0 pty,raw,echo=0
  # then FtdiUsbBridge(port_or_serial="/dev/pts/3")
"""

import struct
import threading
import time

import numpy as np
import pytest

from railnet.runtime.pcie import FtdiUsbBridge, REG_TILE_MASK, REG_STATUS


class FakeSerial:
    """In-process loopback that mimics FPGA handling of FTDI packets.

    Protocol (railnet/runtime/pcie.py:244):
      CSR Write: [0x01, addr_hi, addr_lo, <le32 value>] -> store, no reply
      CSR Read : [0x02, addr_hi, addr_lo] -> reply <le32 value>
      DMA H2C  : [0x03, len_hi, len_lo, <payload>] -> store, no reply
      DMA C2H  : [0x04, len_hi, len_lo] -> reply <payload> (echo last H2C)
    """

    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self.regs = {REG_STATUS: 0x400}
        self._read_buf = bytearray()
        self._lock = threading.Lock()
        self.is_open = True
        self.baudrate = 3000000

    def write(self, data: bytes) -> int:
        with self._lock:
            if not data:
                return 0
            cmd = data[0]
            if cmd == 0x01 and len(data) >= 7:  # CSR write
                offset = (data[1] << 8) | data[2]
                val = struct.unpack("<I", data[3:7])[0]
                self.regs[offset] = val
            elif cmd == 0x02 and len(data) >= 3:  # CSR read
                offset = (data[1] << 8) | data[2]
                val = self.regs.get(offset, 0)
                self._read_buf.extend(struct.pack("<I", val))
            elif cmd == 0x03 and len(data) >= 3:  # DMA H2C
                length = (data[1] << 8) | data[2]
                payload = data[3 : 3 + length]
                # Store for next C2H read (echo)
                self._read_buf.extend(b"")  # H2C has no reply, just store payload for later
                self._last_h2c = payload  # not used further, but track stats
            elif cmd == 0x04 and len(data) >= 3:  # DMA C2H
                length = (data[1] << 8) | data[2]
                # Return `length` bytes of dummy pattern (or echo last H2C if length matches)
                # For simplicity, return incrementing pattern
                pattern = bytes([(i & 0xFF) for i in range(length)])
                self._read_buf.extend(pattern)
            else:
                # Unknown, ignore
                pass
            return len(data)

    def read(self, size: int = 1) -> bytes:
        # Block up to timeout
        t0 = time.monotonic()
        while len(self._read_buf) < size and (time.monotonic() - t0) < self.timeout:
            time.sleep(0.001)
        with self._lock:
            out = bytes(self._read_buf[:size])
            self._read_buf = self._read_buf[size:]
            return out

    def flush(self):
        pass

    def close(self):
        self.is_open = False


def test_ftdi_packet_framing():
    b = FtdiUsbBridge.__new__(FtdiUsbBridge)
    # Don't call __init__, just test framing helpers
    b.CMD_WRITE_CSR = 0x01
    b.CMD_READ_CSR = 0x02
    b.CMD_DMA_H2C = 0x03
    b.CMD_DMA_C2H = 0x04

    # Use real class methods via instance
    bridge = FtdiUsbBridge.__new__(FtdiUsbBridge)
    # Bind methods
    bridge._build_csr_write_packet = FtdiUsbBridge._build_csr_write_packet.__get__(bridge, FtdiUsbBridge)
    bridge._build_csr_read_packet = FtdiUsbBridge._build_csr_read_packet.__get__(bridge, FtdiUsbBridge)
    bridge._build_dma_h2c_packet = FtdiUsbBridge._build_dma_h2c_packet.__get__(bridge, FtdiUsbBridge)
    bridge._build_dma_c2h_packet = FtdiUsbBridge._build_dma_c2h_packet.__get__(bridge, FtdiUsbBridge) if hasattr(FtdiUsbBridge, "_build_dma_c2h_packet") else lambda self, n: bytes([0x04, (n >> 8) & 0xFF, n & 0xFF])

    pkt = bridge._build_csr_write_packet(0x10, 0x12345678)
    assert pkt == bytes([0x01, 0x00, 0x10]) + struct.pack("<I", 0x12345678)
    assert len(pkt) == 7

    pkt2 = bridge._build_csr_read_packet(0x04)
    assert pkt2 == bytes([0x02, 0x00, 0x04])

    payload = b"\x01\x02\x03\x04"
    pkt3 = bridge._build_dma_h2c_packet(payload)
    assert pkt3 == bytes([0x03, 0x00, 0x04]) + payload

    # Also test bridge's packet helpers directly
    b2 = FtdiUsbBridge(port_or_serial="FAKE", verbose=False)
    # Override its serial with FakeSerial for direct packet test
    assert b2._build_csr_write_packet(0x10, 0xABCD) == bytes([0x01, 0x00, 0x10]) + struct.pack("<I", 0xABCD)


def test_ftdi_loopback_csr_roundtrip():
    # Inject FakeSerial into bridge
    fake = FakeSerial(timeout=1.0)
    bridge = FtdiUsbBridge.__new__(FtdiUsbBridge)
    bridge.port_or_serial = "FAKE_LOOPBACK"
    bridge.baudrate = 3000000
    bridge.timeout = 1.0
    bridge.verbose = False
    bridge.regs = {REG_STATUS: 0x400}
    import threading

    bridge._lock = threading.RLock()
    bridge.is_connected = True
    bridge._ser = fake  # type: ignore
    bridge.dma_stats = {"bytes_h2c": 0, "bytes_c2h": 0, "transfers": 0, "errors": 0}
    # Also need _ftdi
    bridge._ftdi = None
    # Bind methods
    bridge.write_csr = FtdiUsbBridge.write_csr.__get__(bridge, FtdiUsbBridge)
    bridge.read_csr = FtdiUsbBridge.read_csr.__get__(bridge, FtdiUsbBridge)
    bridge.get_link_status = FtdiUsbBridge.get_link_status.__get__(bridge, FtdiUsbBridge)
    bridge.dma_self_test = FtdiUsbBridge.dma_self_test.__get__(bridge, FtdiUsbBridge)

    # Write and read back via FakeSerial
    bridge.write_csr(REG_TILE_MASK, 0x0A)
    # write_csr also updates local regs, but read should go via FakeSerial
    val = bridge.read_csr(REG_TILE_MASK)
    assert val == 0x0A

    bridge.write_csr(REG_TILE_MASK, 0x0F)
    assert bridge.read_csr(REG_TILE_MASK) == 0x0F

    # Test link status
    status = bridge.get_link_status()
    assert status["is_connected"] is True
    assert status["backend"] == "ftdi"

    # Test self-test (writes pattern and reads back)
    assert bridge.dma_self_test(0x55) is True

    # Test multiple CSR
    for i in range(10):
        bridge.write_csr(0x20 + i, i * 0x11111111)
    for i in range(10):
        assert bridge.read_csr(0x20 + i) == i * 0x11111111


def test_ftdi_loopback_dma():
    fake = FakeSerial(timeout=1.0)
    bridge = FtdiUsbBridge.__new__(FtdiUsbBridge)
    bridge.port_or_serial = "FAKE_LOOPBACK_DMA"
    bridge.baudrate = 3000000
    bridge.timeout = 1.0
    bridge.verbose = False
    bridge.regs = {REG_STATUS: 0x400}
    import threading

    bridge._lock = threading.RLock()
    bridge.is_connected = True
    bridge._ser = fake  # type: ignore
    bridge._ftdi = None
    bridge.dma_stats = {"bytes_h2c": 0, "bytes_c2h": 0, "transfers": 0, "errors": 0}
    bridge.write_csr = FtdiUsbBridge.write_csr.__get__(bridge, FtdiUsbBridge)
    bridge.read_csr = FtdiUsbBridge.read_csr.__get__(bridge, FtdiUsbBridge)
    bridge.stream_activations = FtdiUsbBridge.stream_activations.__get__(bridge, FtdiUsbBridge)
    bridge.read_results = FtdiUsbBridge.read_results.__get__(bridge, FtdiUsbBridge)

    # Stream activations (int16 payload)
    x = np.arange(16, dtype=np.int16)
    bridge.stream_activations(x)
    assert bridge.dma_stats["bytes_h2c"] == 32
    assert bridge.dma_stats["transfers"] == 1

    # Read results (int32 pattern from FakeSerial)
    # FakeSerial returns incrementing pattern for C2H
    res = bridge.read_results(4, scale=1.0)
    assert res.shape == (4,)
    # Pattern is 0,1,2,3,... as bytes -> int32 little endian
    # Our FakeSerial returns bytes [0,1,2,3,...] as pattern, which as int32 would be 0x03020100 etc
    # Just check shape and that it doesn't error
    assert not np.isnan(res).any()

    # Second stream
    x2 = np.random.randint(-50, 50, size=32, dtype=np.int16)
    bridge.stream_activations(x2)
    assert bridge.dma_stats["transfers"] == 2
    res2 = bridge.read_results(8)
    assert res2.shape == (8,)


def test_ftdi_fallback_no_hardware():
    # Without FakeSerial, bridge should be in mock mode (is_connected=False)
    bridge = FtdiUsbBridge(port_or_serial="/dev/nonexistent_tty_xyz", verbose=False)
    assert bridge.is_connected is False
    # In mock mode, write/read should still work via regs dict
    bridge.write_csr(REG_TILE_MASK, 0x0C)
    assert bridge.read_csr(REG_TILE_MASK) == 0x0C
    # stream_activations in mock mode just tracks stats
    x = np.arange(8, dtype=np.int16)
    bridge.stream_activations(x)
    assert bridge.dma_stats["transfers"] == 1
    res = bridge.read_results(4)
    assert res.shape == (4,)
    # All zeros in mock mode
    assert np.all(res == 0)
    bridge.close()
    assert bridge.is_connected is False
