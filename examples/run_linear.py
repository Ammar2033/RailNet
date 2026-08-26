"""Run linear via rail kernel (toy)."""
import numpy as np
from railnet.compiler import RailNetCompiler
from railnet.dtypes.bf16 import fp32_array_to_bf16_bits
from railnet.kernel import CompiledTensor, rail_linear

# fake 4x2 weight encoded as BF16 bits
W = np.random.randn(4, 2).astype(np.float32)
raw = fp32_array_to_bf16_bits(W.reshape(-1))

comp = RailNetCompiler().compile_tensor(raw, rails=16, max_terms=2, exact=False)
# Build CompiledTensor shim from compiled dict for demo
print("compiled:", comp["exact"], "/", comp["unique"])
