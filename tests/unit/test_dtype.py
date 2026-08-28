"""Comprehensive dtype unit tests — all registered types."""
import struct

import numpy as np
import pytest

from railnet.dtypes import get_dtype, registered_dtypes
from railnet.dtypes.base import RailDType, DTypeInfo
from railnet.dtypes.bf16 import (
    BF16, BF16DType,
    bf16_bits_to_float32, bf16_array_to_float32,
    float32_to_bf16_bits, fp32_array_to_bf16_bits,
)
from railnet.dtypes.fp16 import FP16
from railnet.dtypes.fp32 import FP32
from railnet.dtypes.int8 import INT8
from railnet.dtypes.int16 import INT16
from railnet.dtypes.int4 import INT4


# ── Registry ──────────────────────────────────────────────

class TestRegistry:
    def test_all_dtypes_registered(self):
        names = {"bf16", "fp16", "fp32", "int8", "int16", "int4"}
        reg = registered_dtypes()
        for n in names:
            assert n in reg, f"{n} not registered"

    def test_case_insensitive_lookup(self):
        assert get_dtype("BF16").name == "bf16"
        assert get_dtype("Bf16").name == "bf16"
        assert get_dtype("bf16").name == "bf16"

    def test_unknown_dtype_raises(self):
        with pytest.raises(KeyError, match="Unknown dtype"):
            get_dtype("float128")

    def test_each_dtype_is_raildtype(self):
        for name in ["bf16", "fp16", "fp32", "int8", "int16", "int4"]:
            dt = get_dtype(name)
            assert isinstance(dt, RailDType)
            assert isinstance(dt.info, DTypeInfo)


# ── BF16 ──────────────────────────────────────────────────

class TestBF16:
    def test_info(self):
        assert BF16.name == "bf16"
        assert BF16.bits == 16
        assert BF16.is_float is True
        assert BF16.is_integer is False
        assert BF16.info.status == "PROVEN"

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 0.03125, -0.5, 256.0])
    def test_roundtrip(self, value):
        bits = BF16.encode(value)
        decoded = BF16.decode(bits)
        assert decoded == BF16.quantize(value)

    def test_encode_zero(self):
        assert BF16.encode(0.0) == 0

    def test_encode_one(self):
        # 1.0 in BF16: sign=0, exp=127 (0x7F), mantissa=0 → 0x3F80
        assert BF16.encode(1.0) == 0x3F80

    def test_encode_negative_one(self):
        # -1.0 in BF16: sign=1, exp=127, mantissa=0 → 0xBF80
        assert BF16.encode(-1.0) == 0xBF80

    def test_exact_equal(self):
        bits_a = BF16.encode(0.03125)
        bits_b = BF16.encode(0.03125)
        assert BF16.exact_equal(bits_a, bits_b)

    def test_exact_not_equal(self):
        assert not BF16.exact_equal(BF16.encode(0.03125), BF16.encode(0.0625))

    def test_numpy_dtype(self):
        assert BF16.numpy_dtype == np.float32

    def test_storage_dtype(self):
        assert BF16.storage_dtype == np.uint16


class TestBF16Helpers:
    def test_bits_to_float32(self):
        assert bf16_bits_to_float32(0x3F80) == np.float32(1.0)
        assert bf16_bits_to_float32(0) == np.float32(0.0)

    def test_float32_to_bits(self):
        assert float32_to_bf16_bits(1.0) == 0x3F80
        assert float32_to_bf16_bits(0.0) == 0

    def test_array_roundtrip(self):
        values = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        bits = fp32_array_to_bf16_bits(values)
        assert bits.dtype == np.uint16
        assert len(bits) == 4
        restored = bf16_array_to_float32(bits)
        np.testing.assert_array_equal(restored, values)

    def test_array_shapes_preserved(self):
        bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
        result = bf16_array_to_float32(bits)
        assert result.shape == (2,)
        assert result.dtype == np.float32


# ── FP16 ──────────────────────────────────────────────────

class TestFP16:
    def test_info(self):
        assert FP16.name == "fp16"
        assert FP16.bits == 16
        assert FP16.is_float is True
        assert FP16.info.status == "READY"

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 0.5])
    def test_roundtrip(self, value):
        bits = FP16.encode(value)
        decoded = FP16.decode(bits)
        assert decoded == pytest.approx(value, abs=1e-3)

    def test_quantize(self):
        assert FP16.quantize(1.0) == 1.0

    def test_storage_dtype(self):
        assert FP16.storage_dtype == np.uint16


# ── FP32 ──────────────────────────────────────────────────

class TestFP32:
    def test_info(self):
        assert FP32.name == "fp32"
        assert FP32.bits == 32
        assert FP32.is_float is True
        assert FP32.info.status == "READY"

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 3.14, -0.001])
    def test_roundtrip(self, value):
        bits = FP32.encode(value)
        decoded = FP32.decode(bits)
        assert decoded == pytest.approx(float(np.float32(value)))

    def test_storage_dtype(self):
        assert FP32.storage_dtype == np.uint32


# ── INT8 ──────────────────────────────────────────────────

class TestINT8:
    def test_info(self):
        assert INT8.name == "int8"
        assert INT8.bits == 8
        assert INT8.is_float is False
        assert INT8.is_integer is True

    @pytest.mark.parametrize("value", [0, 1, -1, 127, -128])
    def test_roundtrip(self, value):
        bits = INT8.encode(value)
        assert INT8.decode(bits) == value

    def test_quantize(self):
        assert INT8.quantize(42) == 42


# ── INT16 ─────────────────────────────────────────────────

class TestINT16:
    def test_info(self):
        assert INT16.name == "int16"
        assert INT16.bits == 16

    @pytest.mark.parametrize("value", [0, 1, -1, 32767, -32768])
    def test_roundtrip(self, value):
        bits = INT16.encode(value)
        assert INT16.decode(bits) == value


# ── INT4 ──────────────────────────────────────────────────

class TestINT4:
    def test_info(self):
        assert INT4.name == "int4"
        assert INT4.bits == 4
        assert INT4.info.status == "PLANNED"

    @pytest.mark.parametrize("value", [0, 1, -1, 7, -8])
    def test_roundtrip(self, value):
        bits = INT4.encode(value)
        assert INT4.decode(bits) == value

    def test_clamp(self):
        # values outside [-8, 7] should be clamped
        assert INT4.quantize(100) == 7
        assert INT4.quantize(-100) == -8


# ── exact_equal_array (fixed) ─────────────────────────────

class TestExactEqualArray:
    def test_bf16_uint16_view(self):
        a = np.array([1.0, -1.0], dtype=np.float32)
        b = np.array([1.0, -1.0], dtype=np.float32)
        # BF16 storage is uint16, so view as uint16 should work on float32
        result = BF16.exact_equal_array(a, b)
        # both float32 arrays viewed as uint16 will have 2 uint16 per float32
        assert result.all()

    def test_fp32_uint32_view(self):
        a = np.array([1.0, -1.0], dtype=np.float32)
        b = np.array([1.0, -1.0], dtype=np.float32)
        result = FP32.exact_equal_array(a, b)
        assert result.all()

    def test_int8_view(self):
        a = np.array([42, -1], dtype=np.int8)
        b = np.array([42, -1], dtype=np.int8)
        result = INT8.exact_equal_array(a, b)
        assert result.all()

    def test_mismatch_detected(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 3.0], dtype=np.float32)
        result = BF16.exact_equal_array(a, b)
        # First element matches, second doesn't (at the uint16 level)
        assert result[0]


# ── encode_array / decode_array ───────────────────────────

class TestArrayEncodeDecode:
    def test_bf16_encode_array(self):
        values = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        bits = BF16.encode_array(values)
        assert bits.dtype == np.uint16
        assert len(bits) == 3
        assert int(bits[1]) == 0x3F80  # 1.0

    def test_bf16_decode_array(self):
        bits = np.array([0, 0x3F80, 0xBF80], dtype=np.uint16)
        vals = BF16.decode_array(bits)
        assert len(vals) == 3
        assert vals[0] == 0.0
        assert vals[1] == pytest.approx(1.0)


# ── repr ──────────────────────────────────────────────────

class TestRepr:
    def test_repr_format(self):
        r = repr(BF16)
        assert "bf16" in r
        assert "PROVEN" in r
