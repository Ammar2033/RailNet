"""
Rail — dtype-independent primitive.

Rail = shared numeric value used by many weights:
  W = Σ sign * R[rail_id]   (no per-weight coefficient)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from railnet.dtypes.base import RailDType, get_dtype


@dataclass(frozen=True)
class Rail:
    id: int
    dtype: str  # e.g. "bf16"
    encoded_value: int  # raw bits
    decoded_value: float  # float64 view for compute
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_value(cls, rail_id: int, dtype: str, value: Any, metadata: dict | None = None) -> Rail:
        dt: RailDType = get_dtype(dtype)
        bits = dt.encode(value)
        decoded = dt.decode(bits)
        return cls(
            id=rail_id,
            dtype=dtype,
            encoded_value=int(bits),
            decoded_value=float(decoded),
            metadata=metadata or {},
        )

    @classmethod
    def from_bits(cls, rail_id: int, dtype: str, bits: int, metadata: dict | None = None) -> Rail:
        dt: RailDType = get_dtype(dtype)
        decoded = dt.decode(int(bits))
        return cls(
            id=rail_id,
            dtype=dtype,
            encoded_value=int(bits),
            decoded_value=float(decoded),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dtype": self.dtype,
            "bits": self.encoded_value,
            "value": self.decoded_value,
        }

    def __repr__(self) -> str:
        return f"Rail(id={self.id}, dtype={self.dtype}, bits=0x{self.encoded_value:04X}, value={self.decoded_value})"
