"""Unit tests for storage: MemoryBudget, route_map, honest_report."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from railnet.storage.memory import MemoryBudget
from railnet.storage.route_map import save_route_map, load_route_map, route_map_bytes, honest_report


# ── MemoryBudget ──────────────────────────────────────────

class TestMemoryBudget:
    def test_defaults(self):
        m = MemoryBudget()
        assert m.static_total == 0
        assert m.dynamic_total == 0
        assert m.total == 0

    def test_static(self):
        m = MemoryBudget(static_rails_bytes=192, static_topology_bytes=1000)
        assert m.static_total == 1192

    def test_dynamic(self):
        m = MemoryBudget(dynamic_activation_bytes=4096, dynamic_kv_bytes=2048)
        assert m.dynamic_total == 6144

    def test_total(self):
        m = MemoryBudget(
            static_rails_bytes=100,
            static_topology_bytes=200,
            dynamic_activation_bytes=300,
            dynamic_kv_bytes=400,
        )
        assert m.total == 1000

    def test_to_dict(self):
        m = MemoryBudget(static_rails_bytes=192)
        d = m.to_dict()
        assert d["static_rails_bytes"] == 192
        assert d["static_total"] == 192
        assert "total" in d


# ── Route Map ─────────────────────────────────────────────

class TestRouteMap:
    def test_save_load_roundtrip(self, tmp_path):
        arr = np.array([0, 1, 2, 3, 0, 1], dtype=np.uint16)
        path = str(tmp_path / "route_ids")
        saved = save_route_map(path, arr)
        assert Path(saved).exists()
        loaded = load_route_map(saved)
        np.testing.assert_array_equal(loaded, arr)

    def test_save_converts_to_uint16(self, tmp_path):
        arr = np.array([0, 1, 2], dtype=np.int32)
        path = str(tmp_path / "route.npy")
        saved = save_route_map(path, arr)
        loaded = load_route_map(saved)
        assert loaded.dtype == np.uint16

    def test_route_map_bytes(self):
        # 1000 elements × 16 bits = 2000 bytes
        assert route_map_bytes(1000, 16) == 2000
        # 7 elements × 16 bits = 14 bytes (112 bits / 8)
        assert route_map_bytes(7, 16) == 14


# ── Honest Report ─────────────────────────────────────────

class TestHonestReport:
    def test_small_tensor(self):
        r = honest_report((4, 3), rail_count=4, route_bits=16)
        # dense: 12 * 16 = 192 bits
        assert r["dense_bits"] == 192
        # rail: 4 * 16 = 64 bits
        assert r["rail_bits"] == 64
        # route: 12 * 16 = 192 bits
        assert r["route_bits"] == 192
        # total: 64 + 192 = 256 bits
        assert r["total_bits"] == 256
        # ratio: 192 / 256 = 0.75 (dense is actually cheaper for tiny tensors!)
        assert r["ratio"] == pytest.approx(0.75)

    def test_large_tensor_ratio(self):
        # For large tensors, route_map dominates
        r = honest_report((1152, 6912), rail_count=96, route_bits=16)
        # route cost is dominant, ratio close to 1.0
        assert 0.9 < r["ratio"] < 1.1

    def test_zero_rails(self):
        r = honest_report((4, 3), rail_count=0, route_bits=16)
        assert r["rail_bits"] == 0
