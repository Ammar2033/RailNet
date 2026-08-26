"""
Optimizer — coordinate-descent rail value update.
Thin wrapper over proven update_basis.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent.parent

def _load():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rn_opt_proven", str(_HERE / "04_bf16_learned_basis.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_M = _load()
update_basis = _M.update_basis
repair_missing_values = _M.repair_missing_values
repair_safe_slots = _M.repair_safe_slots
