"""INT8 dtype — READY."""
from __future__ import annotations

import numpy as np

from .base import DTypeInfo, RailDType, register


@register
class INT8DType(RailDType):
    info = DTypeInfo(name="int8", bits=8, is_float=False, is_integer=True, status="READY")
    numpy_dtype = np.int8
    storage_dtype = np.int8

    def encode(self, value) -> int:
        return int(np.int8(value)) & 0xFF

    def decode(self, bits: int):
        v = int(bits) & 0xFF
        if v >= 0x80:
            v -= 0x100
        return v

    def quantize(self, value):
        return int(np.int8(value))


INT8 = INT8DType()
