import numpy as np
from railnet.dtypes import get_dtype

def test_bf16_roundtrip():
    dt = get_dtype("bf16")
    for v in [0.0, 0.03125, -0.5, 1.0]:
        bits = dt.encode(v)
        assert dt.decode(bits) == dt.quantize(v)

def test_fp16():
    dt = get_dtype("fp16")
    assert dt.decode(dt.encode(1.0)) == 1.0

def test_int8():
    dt = get_dtype("int8")
    assert dt.decode(dt.encode(127)) == 127

def test_registry():
    assert get_dtype("BF16").name == "bf16"
