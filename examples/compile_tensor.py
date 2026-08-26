"""Minimal compile_tensor example."""
import numpy as np
from railnet.compiler import RailNetCompiler
from railnet.dtypes.bf16 import fp32_array_to_bf16_bits

# toy weight fragment
w = np.array([0.03125, -0.03125, 0.0625, -0.0625], dtype=np.float32)
raw = fp32_array_to_bf16_bits(w)

compiler = RailNetCompiler(model="generic")
res = compiler.compile_tensor(raw, dtype="bf16", rails=8, max_terms=2, exact=False)
print(res)
