"""FP32 dtype — architecture READY, compiler PLANNED."""

from __future__ import annotations

import struct

import numpy as np

from .base import DTypeInfo, RailDType, register


def fp32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(np.float32(v))))[0]


def bits_to_fp32(b: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", int(b) & 0xFFFFFFFF))[0])


@register
class FP32DType(RailDType):
    info = DTypeInfo(name="fp32", bits=32, is_float=True, is_integer=False, status="READY")
    numpy_dtype = np.float32
    storage_dtype = np.uint32

    def encode(self, value) -> int:
        return fp32_bits(float(value))

    def decode(self, bits: int):
        return bits_to_fp32(int(bits))

    def quantize(self, value):
        return float(np.float32(value))


FP32 = FP32DType()
