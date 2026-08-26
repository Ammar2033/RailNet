"""Runtime linear — delegates to proven kernel.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent.parent
def _load():
    spec = importlib.util.spec_from_file_location("rn_runtime_kernel", str(Path(__file__).resolve().parent.parent / "kernel.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_K = _load()
CompiledTensor = _K.CompiledTensor
rail_linear = _K.rail_linear
prepare = _K.prepare
rail_linear_fast = _K.rail_linear_fast
