"""Router — topology dispatch model."""
from __future__ import annotations

import numpy as np


class Router:
    def __init__(self, rail_count: int, max_terms: int):
        self.rail_count = rail_count
        self.max_terms = max_terms

    def route(self, route_ids: np.ndarray, term_rail: np.ndarray, term_sign: np.ndarray):
        # research stub — real hardware would be crossbar / NoC
        return {"route_ids": route_ids, "term_rail": term_rail, "term_sign": term_sign}
