"""
Generic RailNet compiler — dtype-dispatched.

Usage:
  compiler.compile_tensor(tensor, dtype="bf16", rails=96, max_terms=4, exact=True)
"""
from __future__ import annotations

import time
import numpy as np

from railnet.dtypes import get_dtype
from railnet.rails._analysis import analyze_unique_values
from railnet.rails._optimize import learn_basis
from railnet.rails._compile import compile_exact_routes_exhaustive
from railnet.core import RailTensor, Shape


class RailNetCompiler:
    def __init__(self, model: str = "generic", default_dtype: str = "bf16"):
        self.model = model
        self.default_dtype = default_dtype

    def compile_tensor(self, raw: np.ndarray, dtype: str | None = None, rails: int = 96, max_terms: int = 4, exact: bool = True, name: str = "unknown", shape: tuple | Shape | None = None) -> RailTensor:
        dtype = dtype or self.default_dtype
        dt = get_dtype(dtype)
        if dtype.lower() != "bf16":
            raise NotImplementedError(f"compile_tensor dtype={dtype} is {dt.info.status} — only bf16 PROVEN")

        # raw: uint16 BF16 bits flattened? Accept either float32 or uint16
        if raw.dtype == np.uint16:
            bits_raw = raw.reshape(-1)
        elif raw.dtype == np.float32:
            from railnet.dtypes.bf16 import fp32_array_to_bf16_bits
            bits_raw = fp32_array_to_bf16_bits(raw.reshape(-1))
        else:
            bits_raw = np.asarray(raw, dtype=np.uint16).reshape(-1)

        # learn basis
        bits, counts, vals = analyze_unique_values(bits_raw)
        
        # coordinate descent learning pipeline
        learned = learn_basis(vals, bits, counts, rails, max_terms)
        rails_arr = learned["rails"]
        
        # exhaustive compile
        table = compile_exact_routes_exhaustive(bits, rails_arr, max_terms)
        cov = sum(1 for b in bits if int(b) in table)
        ok = cov == len(bits)
        
        if exact and not ok:
            raise RuntimeError(f"exact compilation failed: {cov}/{len(bits)} with rails={rails}")

        # resolve shape
        if shape is None:
            shape = Shape((len(bits_raw),))
        elif isinstance(shape, tuple):
            shape = Shape(shape)
            
        return RailTensor(
            name=name,
            shape=shape,
            dtype=dtype,
            rail_count=int(rails),
            max_terms=int(max_terms),
            rails_bits=rails_arr,
            routes=table,
            route_ids=bits_raw,
        )

    def compile(self, path: str, dtype: str | None = None, rails: int = 96, max_terms: int = 4):
        """Compile safetensors file -> per-tensor artifacts."""
        return {"status": "use research/experiments/15a_gemma_full_compile.py for bulk model compile", "path": path}
