"""Minimal compile_tensor example — compile one weight fragment into rails + routes."""

import numpy as np

from railnet.compiler import RailNetCompiler
from railnet.dtypes.bf16 import fp32_array_to_bf16_bits

# toy weight fragment (already exact BF16 values)
w = np.array([0.03125, -0.03125, 0.0625, -0.0625], dtype=np.float32)
raw = fp32_array_to_bf16_bits(w)

compiler = RailNetCompiler(model="generic")
tensor = compiler.compile_tensor(raw, dtype="bf16", rails=8, max_terms=2, shape=(2, 2))

print("rails :", [hex(int(b)) for b in tensor.rails_bits])
print("routes:", tensor.to_dict()["routes"])
