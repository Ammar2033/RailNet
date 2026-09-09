"""Unit tests for Gemma-2B Layer-by-Layer Streaming Model Compiler & PCIe Pipeline."""

import numpy as np
import pytest

from railnet.compiler.gemma_streamer import (
    ChunkedLinearCompiler,
    Gemma2BConfig,
    GemmaLayerWeights,
    GemmaStreamingPipeline,
    gelu_approx,
    rms_norm,
)
from railnet.runtime.pcie import RailNetPCIeDriver


def test_gemma_config_and_synthetic_weights():
    """Verify Gemma2BConfig parameters and weight tensor shapes."""
    cfg = Gemma2BConfig()
    assert cfg.hidden_size == 2048
    assert cfg.intermediate_size == 16384
    assert cfg.num_hidden_layers == 18

    # Test micro configuration for fast test execution
    micro_cfg = Gemma2BConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
    )
    weights = GemmaLayerWeights.generate_synthetic(layer_idx=0, config=micro_cfg, seed=123)

    assert weights.q_proj.shape == (32, 32)
    assert weights.k_proj.shape == (8, 32)
    assert weights.v_proj.shape == (8, 32)
    assert weights.gate_proj.shape == (64, 32)
    assert weights.up_proj.shape == (64, 32)
    assert weights.down_proj.shape == (32, 64)


def test_chunked_linear_compiler():
    """Verify matrix partitioning into 4-tile chunks matching FPGA hardware."""
    compiler = ChunkedLinearCompiler(num_tiles=4, rails=16, max_terms=2)
    w = np.random.randn(16, 32).astype(np.float32)

    chunks = compiler.compile_matrix(w)
    assert len(chunks) == 4  # 16 / 4 = 4 chunks

    for i, c in enumerate(chunks):
        assert c.chunk_idx == i
        assert c.out_start == i * 4
        assert c.out_end == (i + 1) * 4
        assert c.compiled_tensor.in_features == 32
        assert c.compiled_tensor.out_features == 4


def test_gemma_activations():
    """Verify GeGLU activation and RMSNorm formulas."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    g = gelu_approx(x)

    # Values should match standard GELU behavior
    assert np.isclose(g[2], 0.0, atol=1e-5)
    assert g[0] < 0.0
    assert g[4] > 1.9

    # RMSNorm test
    w = np.ones(5, dtype=np.float32)
    normed = rms_norm(x, w)
    assert np.isclose(np.mean(np.square(normed)), 1.0, atol=1e-3)


def test_gemma_streaming_mlp_forward():
    """Test chunked tile streaming through Gemma GeGLU MLP block."""
    cfg = Gemma2BConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
    )
    weights = GemmaLayerWeights.generate_synthetic(layer_idx=0, config=cfg, seed=42)

    driver = RailNetPCIeDriver(backend="mock")
    pipeline = GemmaStreamingPipeline(config=cfg, driver=driver, num_tiles=4)

    rng = np.random.default_rng(100)
    x = rng.normal(0.0, 1.0, (cfg.hidden_size,)).astype(np.float32)

    # Compute streamed output
    mlp_out = pipeline.forward_mlp(x, weights)
    assert mlp_out.shape == (cfg.hidden_size,)
    assert not np.isnan(mlp_out).any()
    assert not np.isinf(mlp_out).any()


def test_gemma_full_layer_end_to_end():
    """Test complete single-layer forward pass and verify high correlation with reference."""
    cfg = Gemma2BConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
    )
    weights = GemmaLayerWeights.generate_synthetic(layer_idx=0, config=cfg, seed=999)

    driver = RailNetPCIeDriver(backend="mock")
    pipeline = GemmaStreamingPipeline(config=cfg, driver=driver, num_tiles=4)

    rng = np.random.default_rng(777)
    x = rng.normal(0.0, 1.0, (cfg.hidden_size,)).astype(np.float32)

    streamed_out = pipeline.forward_layer(x, weights)
    ref_out = pipeline.reference_forward_layer(x, weights)

    assert streamed_out.shape == ref_out.shape == (cfg.hidden_size,)

    # Check Pearson correlation between INT8 streamed inference and float32 reference
    corr = float(np.corrcoef(streamed_out, ref_out)[0, 1])
    assert corr > 0.95, f"Expected high correlation with float reference, got {corr:.4f}"
