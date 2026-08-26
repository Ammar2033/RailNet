from railnet.rails import Rail, RailBasis
import numpy as np

def test_rail_from_value():
    r = Rail.from_value(0, "bf16", 0.03125)
    assert r.id == 0
    assert r.dtype == "bf16"

def test_basis():
    bits = np.array([0x3F80, 0xBF80], dtype=np.uint16)
    b = RailBasis.from_bits(bits, dtype="bf16")
    assert len(b) == 2
