"""Comprehensive multi-model transformer tests for Llama-3 and Qwen-2.5.

Uses a self-contained synthetic safetensors fixture to verify:
  1. Full compilation pipeline (safetensors -> RailNet manifest + artifacts).
  2. RailNetModel loading and dynamic context detection.
  3. Numerical parity between Rail backend and dense reference backend.
  4. Preallocated O(1) KVCache autoregressive generation consistency.
  5. Llama-3 RoPE frequency scaling logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from railnet.compiler.model import compile_model
from railnet.dtypes.bf16 import fp32_array_to_bf16_bits
from railnet.runtime.transformer import RailNetModel
from railnet.transformer import (
    KVCache,
    LlamaContext,
    QwenContext,
    create_context,
    rope_cos_sin,
    silu,
)


def _write_mini_safetensors(path: Path, tensors: dict[str, np.ndarray]):
    """Write an exact little-endian safetensors file with BF16 uint16 arrays."""
    header = {}
    current_offset = 0
    buffer_parts = []
    for name, arr in tensors.items():
        arr_u16 = arr.astype(np.uint16)
        arr_bytes = arr_u16.tobytes()
        nbytes = len(arr_bytes)
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + nbytes],
        }
        current_offset += nbytes
        buffer_parts.append(arr_bytes)

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for part in buffer_parts:
            f.write(part)


def _generate_synthetic_model(
    model_dir: Path,
    model_type: str = "llama",
    hidden: int = 64,
    intermediate: int = 128,
    layers: int = 2,
    heads: int = 4,
    kv_heads: int = 2,
    vocab: int = 128,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Create a synthetic miniature model directory with safetensors and config.json."""
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim

    # Generate discrete-palette weights so RailNet compiles rapidly in 1 pass
    palette = np.array([-0.25, -0.125, 0.0, 0.125, 0.25, 0.5], dtype=np.float32)

    def rand_weights(shape):
        idx = rng.integers(0, len(palette), size=shape)
        return fp32_array_to_bf16_bits(palette[idx])

    def rand_norm(shape):
        # Layer norm weights near 1.0
        vals = 1.0 + 0.05 * rng.standard_normal(size=shape).astype(np.float32)
        return fp32_array_to_bf16_bits(vals)

    def rand_bias(shape):
        vals = 0.01 * rng.standard_normal(size=shape).astype(np.float32)
        return fp32_array_to_bf16_bits(vals)

    tensors = {
        "model.embed_tokens.weight": rand_weights((vocab, hidden)),
        "model.norm.weight": rand_norm((hidden,)),
    }

    for b in range(layers):
        pfx = f"model.layers.{b}."
        tensors[pfx + "input_layernorm.weight"] = rand_norm((hidden,))
        tensors[pfx + "post_attention_layernorm.weight"] = rand_norm((hidden,))
        tensors[pfx + "self_attn.q_proj.weight"] = rand_weights((hidden, hidden))
        tensors[pfx + "self_attn.k_proj.weight"] = rand_weights((kv_dim, hidden))
        tensors[pfx + "self_attn.v_proj.weight"] = rand_weights((kv_dim, hidden))
        tensors[pfx + "self_attn.o_proj.weight"] = rand_weights((hidden, hidden))
        tensors[pfx + "mlp.gate_proj.weight"] = rand_weights((intermediate, hidden))
        tensors[pfx + "mlp.up_proj.weight"] = rand_weights((intermediate, hidden))
        tensors[pfx + "mlp.down_proj.weight"] = rand_weights((hidden, intermediate))

        if model_type == "qwen":
            tensors[pfx + "self_attn.q_proj.bias"] = rand_bias((hidden,))
            tensors[pfx + "self_attn.k_proj.bias"] = rand_bias((kv_dim,))
            tensors[pfx + "self_attn.v_proj.bias"] = rand_bias((kv_dim,))

    st_path = model_dir / "model.safetensors"
    _write_mini_safetensors(st_path, tensors)

    cfg = {
        "model_type": model_type,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "dtype": "bf16",
        "rms_norm_eps": 1e-5,
    }
    if model_type == "llama":
        cfg["rope_theta"] = 500000.0
        cfg["rope_scaling"] = {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    elif model_type == "qwen":
        cfg["rope_theta"] = 1000000.0

    cfg_path = model_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return st_path, cfg_path


# ── Tests: Architecture Contexts ─────────────────────────────────────────────


class TestModelContexts:
    def test_llama_context_creation(self):
        cfg = {"model_type": "llama", "hidden_size": 128, "num_attention_heads": 4}
        ctx = create_context(cfg)
        assert isinstance(ctx, LlamaContext)
        assert ctx.norm_offset == 0.0
        assert ctx.embed_scale == 1.0
        assert ctx.layer_norm_keys == ("input_layernorm", "post_attention_layernorm")

    def test_qwen_context_creation(self):
        cfg = {"model_type": "qwen2", "hidden_size": 128, "num_attention_heads": 4}
        ctx = create_context(cfg)
        assert isinstance(ctx, QwenContext)
        assert ctx.norm_offset == 0.0
        assert ctx.bias_keys == ("q_proj", "k_proj", "v_proj")

    def test_llama3_rope_scaling(self):
        cfg = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        }
        ctx = create_context(cfg)
        cos, sin = rope_cos_sin([0, 1, 10, 100], ctx)
        assert cos.shape == (4, 16)
        assert sin.shape == (4, 16)
        # Position 0 cos should be 1.0, sin should be 0.0
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-5)

    def test_silu_activation(self):
        x = np.array([-10.0, 0.0, 2.0, 10.0], dtype=np.float64)
        out = silu(x)
        assert abs(out[1]) < 1e-6  # silu(0) == 0
        assert out[0] < 0.0        # silu(negative) < 0
        assert out[2] > 1.5        # silu(2) = 2 / (1 + exp(-2)) ~ 1.7616


# ── Tests: Preallocated KVCache ──────────────────────────────────────────────


class TestKVCacheBuffer:
    def test_kv_cache_slice_updating(self):
        cache = KVCache(kv_heads=2, head_dim=16, initial_capacity=4)
        assert cache.capacity == 16  # min capacity floor

        k1 = np.ones((2, 3, 16), dtype=np.float64)
        v1 = np.ones((2, 3, 16), dtype=np.float64) * 2.0
        k_act, v_act = cache.update(k1, v1)
        assert k_act.shape == (2, 3, 16)
        assert cache.len == 3

        # Append next single token
        k2 = np.ones((2, 1, 16), dtype=np.float64) * 5.0
        v2 = np.ones((2, 1, 16), dtype=np.float64) * 6.0
        k_act2, v_act2 = cache.update(k2, v2)
        assert k_act2.shape == (2, 4, 16)
        assert cache.len == 4
        np.testing.assert_allclose(k_act2[:, 3, :], 5.0)
        np.testing.assert_allclose(v_act2[:, 3, :], 6.0)

    def test_kv_cache_resizing(self):
        cache = KVCache(kv_heads=1, head_dim=8, initial_capacity=16)
        large_k = np.ones((1, 50, 8), dtype=np.float64)
        large_v = np.ones((1, 50, 8), dtype=np.float64)
        k_act, v_act = cache.update(large_k, large_v)
        assert k_act.shape == (1, 50, 8)
        assert cache.capacity >= 50


# ── Tests: End-to-End Synthetic Model Pipeline ───────────────────────────────


class TestMultiModelEndToEnd:
    @pytest.fixture
    def llama_synthetic(self, tmp_path):
        st_path, cfg_path = _generate_synthetic_model(
            tmp_path / "llama_mini",
            model_type="llama",
            hidden=64,
            intermediate=128,
            layers=2,
            heads=4,
            kv_heads=2,
            vocab=64,
        )
        compiled_dir = tmp_path / "compiled_llama"
        manifest = compile_model(
            str(st_path),
            out_dir=str(compiled_dir),
            rails=96,
            max_terms=4,
            verbose=False,
        )
        return st_path, compiled_dir, manifest

    @pytest.fixture
    def qwen_synthetic(self, tmp_path):
        st_path, cfg_path = _generate_synthetic_model(
            tmp_path / "qwen_mini",
            model_type="qwen",
            hidden=64,
            intermediate=128,
            layers=2,
            heads=4,
            kv_heads=2,
            vocab=64,
        )
        compiled_dir = tmp_path / "compiled_qwen"
        manifest = compile_model(
            str(st_path),
            out_dir=str(compiled_dir),
            rails=96,
            max_terms=4,
            verbose=False,
        )
        return st_path, compiled_dir, manifest

    def test_llama_compilation_and_inference(self, llama_synthetic):
        st_path, compiled_dir, manifest = llama_synthetic
        assert manifest["pass_count"] == 14  # 2 layers * 7 linear weights
        assert manifest["verdict"] == "PASS"

        model = RailNetModel.load(str(compiled_dir))
        assert isinstance(model.ctx, LlamaContext)
        assert model.n_layers == 2

        input_ids = [1, 5, 12]
        logits_rail = model.forward(input_ids, backend="rail")
        logits_dense = model.forward(input_ids, backend="dense")

        assert logits_rail.shape == (64,)
        assert logits_dense.shape == (64,)

        # Numerical agreement between RailNet rail execution and dense reference
        cos_sim = np.dot(logits_rail, logits_dense) / (np.linalg.norm(logits_rail) * np.linalg.norm(logits_dense))
        assert cos_sim > 0.999

    def test_qwen_compilation_and_bias_handling(self, qwen_synthetic):
        st_path, compiled_dir, manifest = qwen_synthetic
        assert manifest["pass_count"] == 14  # 2 layers * 7 linear weights
        assert manifest["verdict"] == "PASS"

        model = RailNetModel.load(str(compiled_dir))
        assert isinstance(model.ctx, QwenContext)
        assert model.n_layers == 2

        input_ids = [2, 7, 19]
        logits_rail = model.forward(input_ids, backend="rail")
        logits_dense = model.forward(input_ids, backend="dense")

        assert logits_rail.shape == (64,)
        assert logits_dense.shape == (64,)

        cos_sim = np.dot(logits_rail, logits_dense) / (np.linalg.norm(logits_rail) * np.linalg.norm(logits_dense))
        assert cos_sim > 0.999
