"""BF16 dtype — PROVEN."""
from __future__ import annotations

import struct

import numpy as np

from .base import DTypeInfo, RailDType, register


def bf16_bits_to_float32(bits: int) -> np.float32:
    fp32_bits = int(bits) << 16
    return np.float32(struct.unpack("<f", struct.pack("<I", fp32_bits))[0])


def bf16_array_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint16(16)).view(np.float32)


def float32_to_bf16_bits(value: float) -> int:
    fp32_bits = struct.unpack("<I", struct.pack("<f", float(np.float32(value))))[0]
    return int(fp32_bits >> 16)


def fp32_array_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def bf16_bitwise_equal(a: int, b: int) -> bool:
    return int(a) == int(b)


@register
class BF16DType(RailDType):
    info = DTypeInfo(name="bf16", bits=16, is_float=True, is_integer=False, status="PROVEN")
    numpy_dtype = np.float32  # compute in fp32 / fp64, store as uint16 bits
    storage_dtype = np.uint16

    def encode(self, value) -> int:
        return float32_to_bf16_bits(float(value))

    def decode(self, bits: int):
        return float(bf16_bits_to_float32(int(bits)))

    def quantize(self, value):
        return self.decode(self.encode(value))

    def exact_equal(self, a_bits: int, b_bits: int) -> bool:
        return int(a_bits) == int(b_bits)


BF16 = BF16DType()
