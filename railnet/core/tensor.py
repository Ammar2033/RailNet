from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shape import Shape


@dataclass
class RailTensor:
    """
    Compiled tensor view (no dense weights).
    Holds rail basis bits + topology table + route_id map handle.
    """

    name: str
    shape: Shape
    dtype: str
    rail_count: int
    max_terms: int
    rails_bits: np.ndarray  # (rail_count,) uint16
    routes: dict[int, tuple]  # bits -> ((rail,sign),...)
    route_ids: np.ndarray | None = None  # (numel,) uint16 per-element map
    metadata: dict = field(default_factory=dict)

    @property
    def numel(self) -> int:
        return self.shape.numel

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape.dims),
            "dtype": self.dtype,
            "rail_count": int(self.rail_count),
            "max_terms": int(self.max_terms),
            "rails": [int(b) for b in self.rails_bits],
            "routes": {
                str(k): [[int(r), int(s)] for r, s in v] for k, v in sorted(self.routes.items())
            },
        }
