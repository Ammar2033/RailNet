"""Unit tests for core module: Shape, RailTensor, RailGraph, RailModel."""

import numpy as np
import pytest

from railnet.core import GraphNode, RailGraph, RailModel, RailTensor, Shape

# ── Shape ─────────────────────────────────────────────────


class TestShape:
    def test_basic(self):
        s = Shape((3, 4))
        assert s.rank == 2
        assert s.numel == 12

    def test_1d(self):
        s = Shape((10,))
        assert s.rank == 1
        assert s.numel == 10

    def test_3d(self):
        s = Shape((2, 3, 4))
        assert s.rank == 3
        assert s.numel == 24

    def test_frozen(self):
        s = Shape((3, 4))
        with pytest.raises(AttributeError):
            s.dims = (5, 6)

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="invalid shape"):
            Shape((0, 4))

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="invalid shape"):
            Shape((-1, 4))

    def test_iter(self):
        s = Shape((3, 4, 5))
        assert list(s) == [3, 4, 5]

    def test_repr(self):
        s = Shape((2, 3))
        assert "Shape" in repr(s)
        assert "2" in repr(s)


# ── RailTensor ────────────────────────────────────────────


class TestRailTensor:
    def _make_tensor(self):
        return RailTensor(
            name="test.weight",
            shape=Shape((4, 3)),
            dtype="bf16",
            rail_count=2,
            max_terms=2,
            rails_bits=np.array([0x3F80, 0xBF80], dtype=np.uint16),
            routes={0x3F80: ((0, 1),), 0xBF80: ((1, -1),)},
        )

    def test_basic(self):
        t = self._make_tensor()
        assert t.name == "test.weight"
        assert t.numel == 12

    def test_to_dict(self):
        t = self._make_tensor()
        d = t.to_dict()
        assert d["name"] == "test.weight"
        assert d["shape"] == [4, 3]
        assert d["dtype"] == "bf16"
        assert d["rail_count"] == 2
        assert d["max_terms"] == 2
        assert len(d["rails"]) == 2
        assert isinstance(d["routes"], dict)

    def test_route_ids_optional(self):
        t = self._make_tensor()
        assert t.route_ids is None

    def test_with_route_ids(self):
        t = RailTensor(
            name="t",
            shape=Shape((4,)),
            dtype="bf16",
            rail_count=2,
            max_terms=1,
            rails_bits=np.array([0x3F80, 0xBF80], dtype=np.uint16),
            routes={},
            route_ids=np.array([0, 1, 0, 1], dtype=np.uint16),
        )
        assert t.route_ids is not None
        assert len(t.route_ids) == 4


# ── RailGraph ─────────────────────────────────────────────


class TestRailGraph:
    def test_add_node(self):
        g = RailGraph()
        n = GraphNode(name="layer0.q_proj", op="rail_linear", inputs=["h"])
        g.add_node(n)
        assert "layer0.q_proj" in g.nodes

    def test_add_edge(self):
        g = RailGraph()
        g.add_edge("input", "layer0.q_proj")
        assert ("input", "layer0.q_proj") in g.edges

    def test_to_dict(self):
        g = RailGraph()
        n = GraphNode(name="op1", op="linear", inputs=["x"], attrs={"dim": 64})
        g.add_node(n)
        g.add_edge("x", "op1")
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert d["nodes"]["op1"]["op"] == "linear"


# ── RailModel ─────────────────────────────────────────────


class TestRailModel:
    def test_basic(self):
        m = RailModel(name="test", architecture="gemma3", dtype="bf16")
        assert m.name == "test"
        assert len(m.tensors) == 0

    def test_add_tensor(self):
        m = RailModel(name="test", architecture="gemma3", dtype="bf16")
        t = RailTensor(
            name="w1",
            shape=Shape((4, 3)),
            dtype="bf16",
            rail_count=2,
            max_terms=2,
            rails_bits=np.array([0x3F80], dtype=np.uint16),
            routes={},
        )
        m.add_tensor(t)
        assert "w1" in m
        assert "w2" not in m

    def test_to_manifest(self):
        m = RailModel(name="test", architecture="gemma3", dtype="bf16")
        t = RailTensor(
            name="w1",
            shape=Shape((4, 3)),
            dtype="bf16",
            rail_count=2,
            max_terms=2,
            rails_bits=np.array([0x3F80], dtype=np.uint16),
            routes={},
        )
        m.add_tensor(t)
        manifest = m.to_manifest()
        assert manifest["model"] == "test"
        assert manifest["architecture"] == "gemma3"
        assert len(manifest["tensors"]) == 1
