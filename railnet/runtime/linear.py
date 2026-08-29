"""Runtime linear — re-exports the proven rail kernel."""

from __future__ import annotations

from railnet.kernel import CompiledTensor, prepare, rail_linear, rail_linear_fast

__all__ = ["CompiledTensor", "prepare", "rail_linear", "rail_linear_fast"]
