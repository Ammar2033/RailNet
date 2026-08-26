"""Rail array — physical lane model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RailArray:
    count: int
    dtype: str = "bf16"
    bits: np.ndarray | None = None  # (count,) uint16

    def program(self, bits: np.ndarray):
        assert len(bits) == self.count, f"rail count mismatch {len(bits)} != {self.count}"
        self.bits = bits.astype(np.uint16)

    def value(self, idx: int) -> int:
        return int(self.bits[idx]) if self.bits is not None else 0
