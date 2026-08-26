"""
Topology graph — maps bit-pattern -> route.

Storage-efficient representation keeps only the unique-value
dictionary; per-element route_ids are stored separately as
.topology / route map file.
"""
from __future__ import annotations

import numpy as np


class TopologyGraph:
    def __init__(self, table: dict[int, tuple], rail_count: int, max_terms: int):
        self.table = dict(table)  # bits -> ((rail,sign),...)
        self.rail_count = int(rail_count)
        self.max_terms = int(max_terms)

    def lookup(self, bits: int):
        return self.table.get(int(bits))

    def __contains__(self, bits: int):
        return int(bits) in self.table

    def __len__(self):
        return len(self.table)

    def coverage(self, unique_bits: np.ndarray) -> int:
        return sum(1 for b in unique_bits if int(b) in self.table)

    def to_serializable(self) -> dict:
        return {str(k): [[int(r), int(s)] for r, s in v] for k, v in sorted(self.table.items())}

    @classmethod
    def from_serializable(cls, d: dict, rail_count: int, max_terms: int):
        table = {int(k): tuple((int(r), int(s)) for r, s in v) for k, v in d.items()}
        return cls(table, rail_count, max_terms)

    def encode_route_ids(self, flat_bits: np.ndarray) -> np.ndarray:
        """Map flat_bits array to topology indices via bit-pattern lookup.
        Requires caller to maintain stable ordering of sorted unique keys.
        Simple helper: returns per-element position in sorted table keys.
        """
        keys = np.array(sorted(self.table.keys()), dtype=np.uint16)
        lut = {int(b): i for i, b in enumerate(keys)}
        return np.array([lut[int(b)] for b in flat_bits], dtype=np.uint16)
