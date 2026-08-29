"""INT16 dtype — READY."""

from __future__ import annotations

import numpy as np

from .base import DTypeInfo, RailDType, register


@register
class INT16DType(RailDType):
    info = DTypeInfo(name="int16", bits=16, is_float=False, is_integer=True, status="READY")
    numpy_dtype = np.int16
    storage_dtype = np.int16

    def encode(self, value) -> int:
        return int(np.int16(value)) & 0xFFFF

    def decode(self, bits: int):
        v = int(bits) & 0xFFFF
        if v >= 0x8000:
            v -= 0x10000
        return v

    def quantize(self, value):
        return int(np.int16(value))


INT16 = INT16DType()
