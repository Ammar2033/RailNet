"""Unit tests for topology: Route, RouteTerm, TopologyGraph, build_schedule."""
import numpy as np
import pytest

from railnet.topology.route import Route, RouteTerm
from railnet.topology.graph import TopologyGraph
from railnet.topology.polarity import polarity_matrix, validate_polarity
from railnet.topology.scheduler import build_schedule


# ── RouteTerm ─────────────────────────────────────────────

class TestRouteTerm:
    def test_valid_positive(self):
        t = RouteTerm(rail=3, sign=1)
        assert t.rail == 3
        assert t.sign == 1

    def test_valid_negative(self):
        t = RouteTerm(rail=17, sign=-1)
        assert t.rail == 17
        assert t.sign == -1

    def test_invalid_sign(self):
        with pytest.raises(ValueError, match="sign must be"):
            RouteTerm(rail=0, sign=0)

    def test_invalid_sign_two(self):
        with pytest.raises(ValueError, match="sign must be"):
            RouteTerm(rail=0, sign=2)

    def test_frozen(self):
        t = RouteTerm(rail=0, sign=1)
        with pytest.raises(AttributeError):
            t.rail = 5


# ── Route ─────────────────────────────────────────────────

class TestRoute:
    def test_from_tuple(self):
        r = Route.from_tuple([(3, 1), (17, -1), (42, 1)])
        assert len(r) == 3
        assert r.terms[0].rail == 3
        assert r.terms[1].sign == -1

    def test_to_tuple(self):
        r = Route.from_tuple([(3, 1), (17, -1)])
        tpl = r.to_tuple()
        assert tpl == ((3, 1), (17, -1))

    def test_roundtrip(self):
        original = [(0, 1), (5, -1)]
        r = Route.from_tuple(original)
        assert Route.from_tuple(r.to_tuple()).to_tuple() == ((0, 1), (5, -1))

    def test_is_empty(self):
        r = Route(terms=[])
        assert r.is_empty()

    def test_not_empty(self):
        r = Route.from_tuple([(0, 1)])
        assert not r.is_empty()

    def test_iterable(self):
        r = Route.from_tuple([(0, 1), (1, -1)])
        terms = list(r)
        assert len(terms) == 2


# ── TopologyGraph ─────────────────────────────────────────

class TestTopologyGraph:
    def _make_graph(self):
        # Two routes: bits 0x3F80 -> [(0, 1)], bits 0xBF80 -> [(1, -1)]
        table = {
            0x3F80: ((0, 1),),
            0xBF80: ((1, -1),),
        }
        return TopologyGraph(table, rail_count=2, max_terms=1)

    def test_lookup(self):
        g = self._make_graph()
        assert g.lookup(0x3F80) == ((0, 1),)
        assert g.lookup(0xBF80) == ((1, -1),)
        assert g.lookup(0x0000) is None

    def test_contains(self):
        g = self._make_graph()
        assert 0x3F80 in g
        assert 0x0000 not in g

    def test_len(self):
        g = self._make_graph()
        assert len(g) == 2

    def test_coverage(self):
        g = self._make_graph()
        bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
        assert g.coverage(bits) == 2

    def test_coverage_partial(self):
        g = self._make_graph()
        bits = np.array([0x3F80, 0x0000], dtype=np.uint16)
        assert g.coverage(bits) == 1

    def test_to_serializable(self):
        g = self._make_graph()
        d = g.to_serializable()
        assert isinstance(d, dict)
        # keys are string
        assert all(isinstance(k, str) for k in d.keys())

    def test_from_serializable(self):
        g = self._make_graph()
        d = g.to_serializable()
        g2 = TopologyGraph.from_serializable(d, rail_count=2, max_terms=1)
        assert len(g2) == len(g)
        assert g2.lookup(0x3F80) == g.lookup(0x3F80)

    def test_encode_route_ids(self):
        g = self._make_graph()
        flat = np.array([0x3F80, 0xBF80, 0x3F80], dtype=np.uint16)
        ids = g.encode_route_ids(flat)
        assert ids.dtype == np.uint16
        assert len(ids) == 3
        # Same bit pattern gets same route id
        assert ids[0] == ids[2]
        assert ids[0] != ids[1]


# ── Polarity ──────────────────────────────────────────────

class TestPolarity:
    def test_polarity_matrix(self):
        signs = np.array([[1, -1], [1, 0]], dtype=np.int8)
        pm = polarity_matrix(signs)
        assert pm.dtype == np.float64
        assert pm[0, 1] == -1.0

    def test_validate_polarity_valid(self):
        signs = np.array([[1, -1], [1, 0]], dtype=np.int8)
        routes = np.array([[1, 1], [1, 0]], dtype=np.int8)
        assert validate_polarity(signs, routes)

    def test_validate_polarity_invalid(self):
        # non-zero sign where route is inactive
        signs = np.array([[1, 1]], dtype=np.int8)
        routes = np.array([[1, 0]], dtype=np.int8)
        assert not validate_polarity(signs, routes)


# ── Scheduler ─────────────────────────────────────────────

class TestBuildSchedule:
    def test_basic_schedule(self):
        # 4 elements, 2 terms, 2 rails, in=2, out=2
        route_ids = np.array([0, 0, 1, 1], dtype=np.uint16)  # 2x2 matrix
        term_rail = np.array([[0, 1], [1, 0]], dtype=np.int32)
        term_sign = np.array([[1, -1], [1, 1]], dtype=np.int8)
        term_active = np.array([[True, True], [True, True]])

        p_ii, p_idx, p_ss = build_schedule(
            route_ids, term_rail, term_sign, term_active,
            in_features=2, out_features=2, rail_count=2,
        )
        assert len(p_ii) > 0
        assert len(p_ii) == len(p_idx) == len(p_ss)

    def test_empty_schedule(self):
        route_ids = np.array([0], dtype=np.uint16)
        term_rail = np.array([[0]], dtype=np.int32)
        term_sign = np.array([[1]], dtype=np.int8)
        term_active = np.array([[False]])  # nothing active

        p_ii, p_idx, p_ss = build_schedule(
            route_ids, term_rail, term_sign, term_active,
            in_features=1, out_features=1, rail_count=1,
        )
        assert len(p_ii) == 0
