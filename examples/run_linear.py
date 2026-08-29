"""Run a linear layer through the RailNet rail kernel and check it against a
dense reference (BF16-bitwise)."""

import numpy as np

from railnet.compiler import RailNetCompiler
from railnet.dtypes.bf16 import bf16_array_to_float32, fp32_array_to_bf16_bits
from railnet.kernel import CompiledTensor, rail_linear

rng = np.random.default_rng(0)
out_features, in_features = 6, 4
W = (rng.standard_normal((out_features, in_features)) * 0.05).astype(np.float32)
raw = fp32_array_to_bf16_bits(W.reshape(-1))

tensor = RailNetCompiler().compile_tensor(
    raw, rails=32, max_terms=4, shape=(out_features, in_features)
)

# Persist the artifact + route map the way CompiledTensor expects, in-memory.
import json
import tempfile
from pathlib import Path

from railnet.artifacts.manifest import checksum_manifest

d = Path(tempfile.mkdtemp())
data = tensor.to_dict()
data["shape"] = [out_features, in_features]
data["checksum_sha256"] = checksum_manifest(data)
(d / "w.json").write_text(json.dumps(data))
comp = CompiledTensor(str(d / "w.json"), tensor.route_ids, (out_features, in_features))

x = (rng.standard_normal(in_features) * 0.1).astype(np.float64)
y_rail = rail_linear(x, comp)
y_dense = x @ bf16_array_to_float32(raw.reshape(out_features, in_features)).astype(np.float64).T

rail_bits = fp32_array_to_bf16_bits(y_rail.astype(np.float32))
dense_bits = fp32_array_to_bf16_bits(y_dense.astype(np.float32))
print("rail  :", y_rail)
print("dense :", y_dense)
print("BF16-bitwise equal:", bool(np.array_equal(rail_bits, dense_bits)))
