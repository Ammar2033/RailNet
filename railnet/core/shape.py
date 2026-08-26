from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    dims: tuple[int, ...]

    def __post_init__(self):
        if any(d <= 0 for d in self.dims):
            raise ValueError(f"invalid shape {self.dims}")

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    def __iter__(self):
        return iter(self.dims)

    def __repr__(self):
        return f"Shape{self.dims}"
