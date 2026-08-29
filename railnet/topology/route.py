"""
Route — per-weight topology.

W = +R3 - R17 + R42  ->  [(3,+1),(17,-1),(42,+1)]
No per-weight floating coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteTerm:
    rail: int
    sign: int  # +1 or -1

    def __post_init__(self):
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")


@dataclass
class Route:
    terms: list[RouteTerm]

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def to_tuple(self):
        return tuple((t.rail, t.sign) for t in self.terms)

    @classmethod
    def from_tuple(cls, tpl):
        return cls([RouteTerm(rail=r, sign=s) for r, s in tpl])

    def is_empty(self) -> bool:
        return len(self.terms) == 0
