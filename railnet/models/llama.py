"""Llama adapter — supporting Llama-3, 3.1, and 3.2 model families."""

from __future__ import annotations

import json
from pathlib import Path

from railnet.compiler import RailNetCompiler

from .base import ModelAdapter

LLAMA_3_2_1B_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "vocab_size": 128256,
    "dtype": "bf16",
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-5,
    "model_type": "llama",
    "tie_word_embeddings": True,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
}

LLAMA_3_2_3B_CONFIG = {
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
    "dtype": "bf16",
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-5,
    "model_type": "llama",
    "tie_word_embeddings": True,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
}


class LlamaAdapter(ModelAdapter):
    name = "llama"
    dtype = "bf16"
    architecture = "llama-3.2-1b"

    def __init__(self, config_path: str | None = None, variant: str = "1b"):
        if variant == "3b":
            self.config = LLAMA_3_2_3B_CONFIG.copy()
            self.architecture = "llama-3.2-3b"
        else:
            self.config = LLAMA_3_2_1B_CONFIG.copy()
            self.architecture = "llama-3.2-1b"

        if config_path and Path(config_path).exists():
            self.config.update(json.loads(Path(config_path).read_text(encoding="utf-8")))

        self.compiler = RailNetCompiler(model="llama", default_dtype="bf16")

    def inspect(self, safetensors_path: str) -> dict:
        from railnet.safetensors_reader import read_header

        hdr, _base = read_header(safetensors_path)
        compilable = [k for k in hdr if any(r in k for r in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))]
        return {
            "tensors": list(hdr.keys())[:10],
            "total": len(hdr),
            "compilable_linears": len(compilable),
            "config": self.config,
        }

    def compile_tensor(self, raw, tensor_name: str, **kwargs):
        rails = kwargs.get("rails", 96)
        terms = kwargs.get("max_terms", 4)
        return self.compiler.compile_tensor(raw, dtype="bf16", rails=rails, max_terms=terms, name=tensor_name)

    def build_graph(self) -> dict:
        return {
            "architecture": self.architecture,
            "layers": self.config["num_hidden_layers"],
            "hidden_size": self.config["hidden_size"],
            "heads": self.config["num_attention_heads"],
            "kv_heads": self.config["num_key_value_heads"],
            "rope_scaling": self.config.get("rope_scaling"),
            "status": "READY",
        }

    def build_runtime(self, compiled_dir: str, device=None):
        from railnet.runtime.transformer import RailNetModel

        return RailNetModel.load(compiled_dir, device=device)
