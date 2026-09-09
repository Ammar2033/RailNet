"""Exactness and end-to-end integration tests for RailNetDevice.sim() cycle-accurate simulator."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from railnet.compiler.model import compile_model
from railnet.dtypes.bf16 import fp32_array_to_bf16_bits
from railnet.runtime.engine import RailNetDevice
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
def compiled_sim_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim_model")
    rng = np.random.default_rng(2026)

    H, I, V = CFG["hidden_size"], CFG["intermediate_size"], CFG["vocab_size"]
    N_HEADS, KV_HEADS, HEAD_DIM = (
        CFG["num_attention_heads"],
        CFG["num_key_value_heads"],
        CFG["head_dim"],
    )
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

    comp_dir = root / "compiled_sim"
    compile_model(
        str(st_path),
        out_dir=str(comp_dir),
        rails=32,
        max_terms=3,
        max_iters=10,
        mixed_precision=True,
        verbose=False,
    )
    return comp_dir


def test_sim_device_numerical_parity(compiled_sim_model):
    """Ensure RailNetDevice.sim produces exact same logits as CPU execution."""
    dev_cpu = RailNetDevice.cpu()
    dev_sim = RailNetDevice.sim(profile="edge_asic")

    model_cpu = RailNetModel.load(str(compiled_sim_model), device=dev_cpu)
    model_sim = RailNetModel.load(str(compiled_sim_model), device=dev_sim)

    prompt_ids = [1, 5, 8]
    logits_cpu = model_cpu.forward(prompt_ids)
    logits_sim = model_sim.forward(prompt_ids)

    np.testing.assert_allclose(logits_sim, logits_cpu, atol=1e-12)


def test_sim_device_telemetry_generation(compiled_sim_model):
    """Ensure simulator records full hardware telemetry across layers."""
    dev_sim = RailNetDevice.sim(profile="edge_asic")
    model_sim = RailNetModel.load(str(compiled_sim_model), device=dev_sim)

    prompt_ids = [2, 4, 6, 8]
    model_sim.forward(prompt_ids)

    tel = dev_sim.get_telemetry(tokens_processed=len(prompt_ids))
    assert tel is not None
    # 4 tokens * (2 layers * 7 linear projections per layer) = 56 linear operations
    assert len(tel.layers) == len(prompt_ids) * 14
    assert tel.total_cycles > 0
    assert tel.total_simulated_ms > 0.0
    assert tel.pcie_dma_latency_ms > 0.0
    assert tel.tokens_per_second > 0.0
    assert tel.average_power_watts > 0.0
    assert tel.average_power_watts < 1.0  # Sub-watt Edge ASIC
    assert tel.tile_utilization_pct > 0.0

    summary_str = tel.summary()
    assert "RailNet Hardware Accelerator Cycle-Accurate Telemetry Report" in summary_str
    assert "Target Profile : Edge ASIC (64 Tiles @ 500 MHz, Gen4x16)" in summary_str
    assert "Sub-watt: True" in summary_str
    assert "layer_0.q_proj" in summary_str
    assert "layer_1.down_proj" in summary_str


def test_sim_device_profiles(compiled_sim_model):
    """Ensure different hardware profiles simulate with expected characteristics."""
    dev_fpga = RailNetDevice.sim(profile="fpga_proto")
    model_fpga = RailNetModel.load(str(compiled_sim_model), device=dev_fpga)
    model_fpga.forward([3, 7])
    tel_fpga = dev_fpga.get_telemetry(tokens_processed=2)

    dev_cloud = RailNetDevice.sim(profile="cloud_asic")
    model_cloud = RailNetModel.load(str(compiled_sim_model), device=dev_cloud)
    model_cloud.forward([3, 7])
    tel_cloud = dev_cloud.get_telemetry(tokens_processed=2)

    # Cloud ASIC runs @ 1000 MHz with 256 tiles vs FPGA @ 150 MHz with 16 tiles
    # Cloud latency should be substantially lower than FPGA prototype
    assert tel_cloud.total_simulated_ms < tel_fpga.total_simulated_ms
    assert tel_cloud.tokens_per_second > tel_fpga.tokens_per_second
