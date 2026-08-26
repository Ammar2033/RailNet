"""Model-level compile orchestration."""
from __future__ import annotations

from pathlib import Path

from .compiler import RailNetCompiler


def compile_model(safetensors_path: str, out_dir: str = "compiled", dtype: str = "bf16"):
    # Delegates to proven 15a script — keeps single source of truth
    import importlib.util, sys
    p = Path(__file__).resolve().parent.parent.parent / "research" / "experiments" / "15a_gemma_full_compile.py"
    spec = importlib.util.spec_from_file_location("rn_model_compile", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "main"):
        return m.main()
    return {"status": "no main in 15a"}
