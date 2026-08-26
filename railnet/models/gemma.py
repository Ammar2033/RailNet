"""Gemma adapter — PROVEN on Gemma3 1B."""
from __future__ import annotations

import json
from pathlib import Path

from .base import ModelAdapter
from railnet.compiler import RailNetCompiler
from railnet.dtypes import get_dtype


GEMMA_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 6912,
    "num_hidden_layers": 26,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "vocab_size": 262144,
    "dtype": "bf16",
    "rope_local_base_freq": 10000.0,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 256,
}


class GemmaAdapter(ModelAdapter):
    name = "gemma3"
    dtype = "bf16"
    architecture = "gemma3-1b"

    def __init__(self, config_path: str | None = None):
        self.config = GEMMA_CONFIG.copy()
        if config_path and Path(config_path).exists():
            self.config.update(json.loads(Path(config_path).read_text()))
        self.compiler = RailNetCompiler(model="gemma3", default_dtype="bf16")

    def inspect(self, safetensors_path: str) -> dict:
        from railnet.safetensors_reader import read_header
        hdr, base = read_header(safetensors_path)
        return {"tensors": list(hdr.keys())[:10], "total": len(hdr), "config": self.config}

    def compile_tensor(self, raw, tensor_name: str, **kwargs):
        rails = kwargs.get("rails", 96)
        terms = kwargs.get("max_terms", 4)
        return self.compiler.compile_tensor(raw, dtype="bf16", rails=rails, max_terms=terms)

    def build_graph(self) -> dict:
        return {"architecture": self.architecture, "layers": self.config["num_hidden_layers"]}

    def build_runtime(self, compiled_dir: str, device=None):
        from railnet.runtime.transformer import RailNetModel
        return RailNetModel.load(compiled_dir, device=device)
