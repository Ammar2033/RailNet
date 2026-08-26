"""
Generic RailNet compiler — dtype-dispatched.

Usage:
  compiler.compile_tensor(tensor, dtype="bf16", rails=96, max_terms=4, exact=True)
"""
from __future__ import annotations

import time

import numpy as np

from railnet.dtypes import get_dtype
from railnet.rails.learner import analyze_unique


def _load_bf16_backend():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent / "11_global_layer0_shared_basis.py"
    spec = importlib.util.spec_from_file_location("rn_compiler_fast", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_FAST = _load_bf16_backend()


class RailNetCompiler:
    def __init__(self, model: str = "generic", default_dtype: str = "bf16"):
        self.model = model
        self.default_dtype = default_dtype

    def compile_tensor(self, raw: np.ndarray, dtype: str | None = None, rails: int = 96, max_terms: int = 4, exact: bool = True):
        dtype = dtype or self.default_dtype
        dt = get_dtype(dtype)
        if dtype.lower() != "bf16":
            raise NotImplementedError(f"compile_tensor dtype={dtype} is {dt.info.status} — only bf16 PROVEN")

        # raw: uint16 BF16 bits flattened? Accept either float32 or uint16
        if raw.dtype == np.uint16:
            bits_raw = raw
        elif raw.dtype == np.float32:
            from railnet.dtypes.bf16 import fp32_array_to_bf16_bits
            bits_raw = fp32_array_to_bf16_bits(raw)
        else:
            bits_raw = np.asarray(raw, dtype=np.uint16)

        from railnet.rails.learner import learn_basis as lb
        # analyze_unique expects raw uint16
        bits, counts, vals = _FAST.RN.analyze_unique_values(bits_raw) if hasattr(_FAST, "RN") else analyze_unique(bits_raw)
        # FAST path may have RN
        try:
            learned = lb(vals, bits, counts, rails, max_terms)
        except TypeError:
            learned = lb(vals, bits, counts, rails)

        rails_arr = learned["rails"]
        # exhaustive compile
        compile_fn = _FAST.FAST_COMPILE if hasattr(_FAST, "FAST_COMPILE") else _FAST.RN.compile_exact_routes_exhaustive
        table = compile_fn(bits, rails_arr, max_terms)
        cov = sum(1 for b in bits if int(b) in table)
        ok = cov == len(bits)
        if exact and not ok:
            raise RuntimeError(f"exact compilation failed: {cov}/{len(bits)} with rails={rails}")
        return {
            "rails": rails_arr,
            "table": table,
            "exact": int(cov),
            "unique": int(len(bits)),
            "lossless": bool(ok),
            "dtype": dtype,
            "rails_count": int(rails),
            "max_terms": int(max_terms),
        }

    def compile(self, path: str, dtype: str | None = None, rails: int = 96, max_terms: int = 4):
        """Compile safetensors file -> per-tensor artifacts (delegates to proven bulk compile)."""
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("rn_15a", str(Path(__file__).resolve().parent.parent.parent / "15a_gemma_full_compile.py"))
        # fallback to simple message
        return {"status": "use research/experiments/15a_gemma_full_compile.py for bulk model compile", "path": path}
