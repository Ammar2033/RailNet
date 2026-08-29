"""INT4 dtype — future research, PLANNED."""

from __future__ import annotations

import numpy as np

from .base import DTypeInfo, RailDType, register


@register
class INT4DType(RailDType):
    info = DTypeInfo(name="int4", bits=4, is_float=False, is_integer=True, status="PLANNED")
    numpy_dtype = np.int8  # stored nibble in int8
    storage_dtype = np.uint8

    def encode(self, value) -> int:
        v = int(value)
        v = max(-8, min(7, v))
        return v & 0xF

    def decode(self, bits: int):
        v = int(bits) & 0xF
        if v >= 8:
            v -= 16
        return v

    def quantize(self, value):
        return self.decode(self.encode(value))


INT4 = INT4DType()
