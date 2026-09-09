"""End-to-end exactness verification for Mixed-Precision (BF16 Attention + INT8 MLP)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from railnet.compiler.model import compile_model
from railnet.dtypes.bf16 import fp32_array_to_bf16_bits
from railnet.runtime.transformer import RailNetModel

CFG = {
    "model_type": "gemma3_text",
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "vocab_size": 24,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 4,
    "rope_local_base_freq": 10000.0,
    "eos_token_id": [23],
}


def _write_safetensors(path, tensors: dict[str, np.ndarray]) -> None:
    header, blob, offset = {}, bytearray(), 0
    for name, arr in tensors.items():
        bits = fp32_array_to_bf16_bits(arr.astype(np.float32)).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(bits)],
        }
        blob += bits
        offset += len(bits)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


@pytest.fixture(scope="module")
def mixed_model_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed_model")
    rng = np.random.default_rng(2026)

    H, I, V = CFG["hidden_size"], CFG["intermediate_size"], CFG["vocab_size"]
    N_HEADS, KV_HEADS, HEAD_DIM = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
    Q_DIM = N_HEADS * HEAD_DIM
    KV_DIM = KV_HEADS * HEAD_DIM

    def w(*shape):
        q = np.round(rng.standard_normal(shape) * 6.0) / 128.0
        return q.astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": w(V, H),
        "model.norm.weight": w(H),
    }
    for b in range(CFG["num_hidden_layers"]):
        tensors[f"model.layers.{b}.input_layernorm.weight"] = w(H)
        tensors[f"model.layers.{b}.post_attention_layernorm.weight"] = w(H)
        tensors[f"model.layers.{b}.pre_feedforward_layernorm.weight"] = w(H)
        tensors[f"model.layers.{b}.post_feedforward_layernorm.weight"] = w(H)

        tensors[f"model.layers.{b}.self_attn.q_proj.weight"] = w(Q_DIM, H)
        tensors[f"model.layers.{b}.self_attn.k_proj.weight"] = w(KV_DIM, H)
        tensors[f"model.layers.{b}.self_attn.v_proj.weight"] = w(KV_DIM, H)
        tensors[f"model.layers.{b}.self_attn.o_proj.weight"] = w(H, Q_DIM)
        tensors[f"model.layers.{b}.self_attn.q_norm.weight"] = w(HEAD_DIM)
        tensors[f"model.layers.{b}.self_attn.k_norm.weight"] = w(HEAD_DIM)

        tensors[f"model.layers.{b}.mlp.gate_proj.weight"] = w(I, H)
        tensors[f"model.layers.{b}.mlp.up_proj.weight"] = w(I, H)
        tensors[f"model.layers.{b}.mlp.down_proj.weight"] = w(H, I)

    st_path = root / "model.safetensors"
    _write_safetensors(st_path, tensors)
    (root / "config.json").write_text(json.dumps(CFG))

    comp_dir = root / "compiled_mixed"
    manifest = compile_model(
        str(st_path),
        out_dir=str(comp_dir),
        rails=32,
        max_terms=3,
        max_iters=10,
        mixed_precision=True,
        verbose=False,
    )
    return comp_dir, manifest


def test_mixed_precision_manifest_structure(mixed_model_dir):
    comp_dir, manifest = mixed_model_dir
    assert manifest["verdict"] == "PASS"
    assert manifest["mixed_precision"] is True
    assert manifest["policy"] == {"mlp": "int8", "attn": "bf16"}

    # Check layer dtypes
    for name, entry in manifest["tensors"].items():
        role = entry["role"]
        if role in ("gate_proj", "up_proj", "down_proj"):
            assert entry["dtype"] == "int8"
            assert entry["scale"] > 0
            assert entry["rails"] <= 32
        else:
            assert entry["dtype"] == "bf16"


def test_mixed_precision_model_loading_and_inference(mixed_model_dir):
    comp_dir, _ = mixed_model_dir

    model_auto = RailNetModel.load(str(comp_dir), backend="auto")
    assert model_auto.is_fully_compiled

    tokens = [2, 7, 15]
    logits_auto = model_auto.forward(tokens)
    assert logits_auto.shape == (CFG["vocab_size"],)
    assert not np.isnan(logits_auto).any()

    # Verify backends agree
    model_numpy = RailNetModel.load(str(comp_dir), backend="numpy")
    logits_numpy = model_numpy.forward(tokens)

    np.testing.assert_allclose(logits_auto, logits_numpy, rtol=1e-3, atol=1e-3)


def test_mixed_precision_greedy_generation(mixed_model_dir):
    comp_dir, _ = mixed_model_dir
    model = RailNetModel.load(str(comp_dir), backend="auto")

    prompt = [3, 8]
    gen_res = model.generate(prompt, max_new_tokens=4)
    assert "tokens" in gen_res
    assert len(gen_res["tokens"]) >= 1
