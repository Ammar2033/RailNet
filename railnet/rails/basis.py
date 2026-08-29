"""
Basis — ordered collection of rails.

Represents the shared fabric:

  R0, R1, ..., R{N-1}

Physical rail capacity is fixed (e.g. 96 or 128).
Values are programmable per model artifact.
"""

from __future__ import annotations

import numpy as np

from railnet.dtypes.base import get_dtype

from .rail import Rail


class RailBasis:
    def __init__(self, rails: list[Rail] | np.ndarray, dtype: str = "bf16"):
        if isinstance(rails, np.ndarray):
            # uint16 bits array
            self.dtype_name = dtype
            get_dtype(dtype)
            self.rails = [Rail.from_bits(i, dtype, int(b)) for i, b in enumerate(rails)]
            self.bits = rails.astype(np.uint16)
            self.values_f64 = np.array([r.decoded_value for r in self.rails], dtype=np.float64)
        else:
            self.rails = list(rails)
            self.dtype_name = self.rails[0].dtype if self.rails else dtype
            self.bits = np.array([r.encoded_value for r in self.rails], dtype=np.uint16)
            self.values_f64 = np.array([r.decoded_value for r in self.rails], dtype=np.float64)

    @classmethod
    def from_bits(cls, bits: np.ndarray, dtype: str = "bf16") -> RailBasis:
        return cls(bits, dtype=dtype)

    def __len__(self):
        return len(self.rails)

    def __getitem__(self, idx):
        return self.rails[idx]

    def to_bits(self) -> np.ndarray:
        return self.bits.copy()

    def to_dict(self) -> dict:
        return {
            "dtype": self.dtype_name,
            "rail_count": len(self.rails),
            "rails": [r.to_dict() for r in self.rails],
        }
