"""
Optimizer — coordinate-descent rail value update.
Thin wrapper over proven update_basis.
"""
from __future__ import annotations

from ._optimize import update_basis
from ._repair import repair_missing_values, repair_safe_slots
