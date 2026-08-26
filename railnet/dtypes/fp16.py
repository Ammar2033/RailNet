"""FP16 dtype — architecture READY, compiler TODO."""
from __future__ import annotations

import numpy as np

from .base import DTypeInfo, RailDType, register


@register
class FP16DType(RailDType):
    info = DTypeInfo(name="fp16", bits=16, is_float=True, is_integer=False, status="READY")
    numpy_dtype = np.float16
    storage_dtype = np.uint16

    def encode(self, value) -> int:
        return int(np.float16(value).view(np.uint16))

    def decode(self, bits: int):
        return float(np.uint16(bits).view(np.float16))

    def quantize(self, value):
        return float(np.float16(value))


FP16 = FP16DType()
