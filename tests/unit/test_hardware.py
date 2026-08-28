"""Unit tests for hardware architecture models."""
import pytest

from railnet.hardware.architecture import RailNetCard, RailFabric, RoutingFabric, ComputeFabric, MemorySubsystem
from railnet.hardware.pci import PCIeDevice, PCIeLink
from railnet.hardware.rail_array import RailArray
from railnet.hardware.router import Router
import numpy as np


# ── RailFabric ────────────────────────────────────────────

class TestRailFabric:
    def test_defaults(self):
        f = RailFabric()
        assert f.rail_count == 96
        assert f.dtype == "bf16"
        assert f.programmable is True

    def test_custom(self):
        f = RailFabric(rail_count=128, dtype="fp16")
        assert f.rail_count == 128


# ── RoutingFabric ─────────────────────────────────────────

class TestRoutingFabric:
    def test_defaults(self):
        r = RoutingFabric()
        assert r.max_terms == 4
        assert r.storage == "SRAM"


# ── RailNetCard ───────────────────────────────────────────

class TestRailNetCard:
    def test_to_dict(self):
        card = RailNetCard()
        d = card.to_dict()
        assert d["pcie"] == "Gen4 x16"
        assert d["status"] == "RESEARCH / NOT YET SILICON"
        assert d["rail_fabric"]["rail_count"] == 96


# ── PCIeDevice ────────────────────────────────────────────

class TestPCIeDevice:
    def test_creation(self):
        dev = PCIeDevice("railnet0")
        assert dev.name == "railnet0"
        assert dev.link.gen == "Gen4"
        assert dev.link.lanes == 16

    def test_dma_write(self):
        dev = PCIeDevice()
        written = dev.dma_write(b"test data")
        assert written == 9

    def test_dma_read(self):
        dev = PCIeDevice()
        data = dev.dma_read(16)
        assert len(data) == 16
        assert data == b"\x00" * 16

    def test_custom_link(self):
        link = PCIeLink(gen="Gen5", lanes=8, bandwidth_gbps=64.0)
        dev = PCIeDevice(link=link)
        assert dev.link.bandwidth_gbps == 64.0


# ── RailArray ─────────────────────────────────────────────

class TestRailArray:
    def test_creation(self):
        ra = RailArray(count=96)
        assert ra.count == 96
        assert ra.bits is None

    def test_program(self):
        ra = RailArray(count=2)
        bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
        ra.program(bits)
        assert ra.value(0) == 0x3F80
        assert ra.value(1) == 0xBF80

    def test_program_count_mismatch(self):
        ra = RailArray(count=2)
        with pytest.raises(AssertionError, match="rail count mismatch"):
            ra.program(np.array([1, 2, 3], dtype=np.uint16))

    def test_value_unprogrammed(self):
        ra = RailArray(count=2)
        assert ra.value(0) == 0


# ── Router ────────────────────────────────────────────────

class TestRouter:
    def test_creation(self):
        r = Router(rail_count=96, max_terms=4)
        assert r.rail_count == 96

    def test_route_passthrough(self):
        r = Router(rail_count=2, max_terms=2)
        ids = np.array([0, 1], dtype=np.uint16)
        tr = np.array([[0, 1]], dtype=np.int32)
        ts = np.array([[1, -1]], dtype=np.int8)
        result = r.route(ids, tr, ts)
        np.testing.assert_array_equal(result["route_ids"], ids)
