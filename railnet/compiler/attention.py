"""Attention compiler — delegates per-projection compile."""
from .linear import compile_linear


def compile_attention_projections(proj_dict: dict, dtype="bf16", rails=96, max_terms=4):
    return {k: compile_linear(v, name=k, dtype=dtype, rails=rails, max_terms=max_terms) for k, v in proj_dict.items()}
