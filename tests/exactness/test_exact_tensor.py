import numpy as np
from railnet.verification.exact import verify_tensor_exact

def test_exact_toy():
    # two rails can represent 4 values with 2 terms
    rails = np.array([0x3D00, 0x3E80], dtype=np.uint16)  # ~0.03125, ~0.25
    table = {0x3D00: ((0, 1),), 0x3E80: ((1, 1),)}
    uniq = np.array([0x3D00, 0x3E80], dtype=np.uint16)
    r = verify_tensor_exact(uniq, table, rails)
    assert r["lossless"]
