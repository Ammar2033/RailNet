"""Qwen adapter — supporting Qwen-2 and Qwen-2.5 model families."""

from __future__ import annotations

import json
from pathlib import Path

from railnet.compiler import RailNetCompiler

from .base import ModelAdapter

QWEN_2_5_0_5B_CONFIG = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 151936,
    "dtype": "bf16",
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "model_type": "qwen2",
    "tie_word_embeddings": True,
}

QWEN_2_5_1_5B_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 151936,
    "dtype": "bf16",
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "model_type": "qwen2",
    "tie_word_embeddings": True,
}


class QwenAdapter(ModelAdapter):
    name = "qwen"
    dtype = "bf16"
    architecture = "qwen-2.5-0.5b"

    def __init__(self, config_path: str | None = None, variant: str = "0.5b"):
        if variant == "1.5b":
            self.config = QWEN_2_5_1_5B_CONFIG.copy()
            self.architecture = "qwen-2.5-1.5b"
        else:
            self.config = QWEN_2_5_0_5B_CONFIG.copy()
            self.architecture = "qwen-2.5-0.5b"

        if config_path and Path(config_path).exists():
            self.config.update(json.loads(Path(config_path).read_text(encoding="utf-8")))

        self.compiler = RailNetCompiler(model="qwen", default_dtype="bf16")

    def inspect(self, safetensors_path: str) -> dict:
        from railnet.safetensors_reader import read_header

        hdr, _base = read_header(safetensors_path)
        compilable = [k for k in hdr if any(r in k for r in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))]
        has_bias = any(".bias" in k for k in hdr)
        return {
            "tensors": list(hdr.keys())[:10],
            "total": len(hdr),
            "compilable_linears": len(compilable),
            "has_qkv_bias": has_bias,
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
            "has_qkv_bias": True,
            "status": "READY",
        }

    def build_runtime(self, compiled_dir: str, device=None):
        from railnet.runtime.transformer import RailNetModel

        return RailNetModel.load(compiled_dir, device=device)
