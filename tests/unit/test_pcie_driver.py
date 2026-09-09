"""Unit tests for RailNet PCIe/FPGA Runtime Driver and MockPCIeBridge."""

import numpy as np
import pytest

from railnet.kernel import CompiledTensor, prepare, rail_linear_fast
from railnet.runtime.engine import RailNetDevice, RailNetEngine
from railnet.runtime.pcie import (
    MockPCIeBridge,
    RailNetPCIeDriver,
    REG_CTRL,
    REG_STATUS,
    REG_IN_FEATURES,
    REG_OUT_FEATURES,
    REG_TILE_MASK,
    REG_CYCLE_COUNT,
)


def _make_dummy_compiled(
    out_features: int = 8,
    in_features: int = 16,
    rail_count: int = 8,
    max_terms: int = 2,
    num_routes: int = 32,
    seed: int = 42,
) -> CompiledTensor:
    """Helper to create a small compiled tensor for hardware testing."""
    rng = np.random.RandomState(seed)

    c = CompiledTensor.__new__(CompiledTensor)
    c.checksum_ok = True
    c.tensor_name = "pcie_dummy_linear"
    c.rail_count = rail_count
    c.max_terms = max_terms
    c.shape = (out_features, in_features)
    c.out_features = out_features
    c.in_features = in_features
    c.load_seconds = 0.0
    c.scale = 1.0

    c.rails_f64 = rng.uniform(-0.5, 0.5, size=rail_count).astype(np.float64)

    c.term_rail = np.zeros((65536, max_terms), dtype=np.int32)
    c.term_sign = np.zeros((65536, max_terms), dtype=np.int8)
    c.term_active = np.zeros((65536, max_terms), dtype=bool)

    for g in range(num_routes):
        num_t = rng.randint(1, max_terms + 1)
        r_indices = rng.choice(rail_count, size=num_t, replace=False)
        signs = rng.choice([-1, 1], size=num_t)
        for t in range(num_t):
            c.term_rail[g, t] = r_indices[t]
            c.term_sign[g, t] = signs[t]
            c.term_active[g, t] = True

    c.route_ids = rng.randint(0, num_routes, size=(out_features, in_features)).astype(np.int32)
    c.prepared = False
    prepare(c)
    return c


def test_mock_pcie_bridge_csr():
    bridge = MockPCIeBridge(num_tiles=4)

    # Status register initial check: num_tiles in [15:8]
    status = bridge.read_csr(REG_STATUS)
    assert (status >> 8) == 4

    # Write & read IN_FEATURES, OUT_FEATURES, TILE_MASK
    bridge.write_csr(REG_IN_FEATURES, 256)
    assert bridge.read_csr(REG_IN_FEATURES) == 256

    bridge.write_csr(REG_OUT_FEATURES, 64)
    assert bridge.read_csr(REG_OUT_FEATURES) == 64

    bridge.write_csr(REG_TILE_MASK, 0x0F)
    assert bridge.read_csr(REG_TILE_MASK) == 0x0F

    # Trigger start bit
    bridge.write_csr(REG_CTRL, 1)
    status = bridge.read_csr(REG_STATUS)
    assert (status & 0x1) == 1  # busy bit set


def test_pcie_driver_fallback_and_csr():
    driver = RailNetPCIeDriver("railnet_test_nonexistent")
    assert not driver.is_hardware
    assert driver.bridge is not None

    driver.write_csr(REG_IN_FEATURES, 128)
    assert driver.read_csr(REG_IN_FEATURES) == 128

    driver.write_csr(REG_OUT_FEATURES, 32)
    assert driver.read_csr(REG_OUT_FEATURES) == 32


def test_pcie_driver_linear_dispatch_1d():
    driver = RailNetPCIeDriver("railnet_test_dev")
    c = _make_dummy_compiled(in_features=16, out_features=8)
    x = np.random.randn(16).astype(np.float64)

    y_hw = driver.dispatch_linear(x, c)
    y_ref = rail_linear_fast(x, c)

    assert y_hw.shape == (8,)
    np.testing.assert_allclose(y_hw, y_ref, rtol=1e-5, atol=1e-5)
    assert driver.get_cycle_count() > 0


def test_pcie_driver_linear_dispatch_2d_batch():
    driver = RailNetPCIeDriver("railnet_test_dev")
    c = _make_dummy_compiled(in_features=24, out_features=12)
    x_batch = np.random.randn(4, 24).astype(np.float64)

    y_hw = driver.dispatch_linear(x_batch, c)
    assert y_hw.shape == (4, 12)

    for i in range(4):
        y_ref = rail_linear_fast(x_batch[i], c)
        np.testing.assert_allclose(y_hw[i], y_ref, rtol=1e-5, atol=1e-5)


def test_pcie_driver_streaming_methods():
    driver = RailNetPCIeDriver("railnet_test_dev")
    c = _make_dummy_compiled(in_features=16, out_features=8)
    x = np.random.randn(16).astype(np.float64)

    driver.bridge.program_tensor(c)
    driver.stream_activations(x)
    results = driver.read_results(8, scale=1.0)

    assert results.shape == (8,)
    y_ref = rail_linear_fast(x, c)
    np.testing.assert_allclose(results, y_ref, rtol=1e-5, atol=1e-5)


def test_engine_pcie_device():
    dev = RailNetDevice.pcie("railnet0")
    assert dev.kind == "pcie"
    assert hasattr(dev, "pcie_driver")

    engine = RailNetEngine(device=dev)
    c = _make_dummy_compiled(in_features=20, out_features=10)
    x = np.random.randn(20).astype(np.float64)

    y_out = engine.dispatch_linear(x, c)
    y_ref = rail_linear_fast(x, c)
    np.testing.assert_allclose(y_out, y_ref, rtol=1e-5, atol=1e-5)

    dev.close()


def test_engine_open_factory():
    dev = RailNetDevice.open("railnet1")
    assert dev.kind == "pcie"
    assert dev.name == "railnet1"
    dev.close()


def test_transformer_model_pcie_device(tmp_path):
    from railnet.compiler.model import compile_model
    from railnet.runtime.transformer import RailNetModel
    from tests.unit.test_transformer_multimodel import _generate_synthetic_model

    st_path, cfg_path = _generate_synthetic_model(
        tmp_path / "llama_mini_pcie",
        model_type="llama",
        hidden=32,
        intermediate=64,
        layers=1,
        heads=2,
        kv_heads=1,
        vocab=32,
    )
    compiled_dir = tmp_path / "compiled_llama_pcie"
    compile_model(
        str(st_path),
        out_dir=str(compiled_dir),
        rails=32,
        max_terms=2,
        verbose=False,
    )

    dev = RailNetDevice.pcie("railnet0")
    model = RailNetModel.load(str(compiled_dir), device=dev)
    assert model.device is dev
    assert model.device.kind == "pcie"

    input_ids = [1, 3]
    logits = model.forward(input_ids, backend="rail")
    assert logits.shape == (32,)
    assert not np.isnan(logits).any()

    logits_ref = model.forward(input_ids, backend="dense")
    cos_sim = np.dot(logits, logits_ref) / (np.linalg.norm(logits) * np.linalg.norm(logits_ref))
    assert cos_sim > 0.99
    dev.close()


def test_pcie_driver_program_layer_all_tiles_and_codebook():
    """Verify that program_layer programs all tiles, codebook, routes, and rails."""
    driver = RailNetPCIeDriver("railnet_test_prog")
    c = _make_dummy_compiled(in_features=16, out_features=4, rail_count=8, num_routes=16)

    # Force driver to execute the real hardware programming sequence on its mock bridge
    driver.is_hardware = True
    driver.bridge = MockPCIeBridge(num_tiles=4)

    # Mock CSR writes directly onto bridge
    driver.write_csr = driver.bridge.write_csr
    driver.read_csr = driver.bridge.read_csr

    driver.program_layer(c)

    # Verify that all 4 tiles received routes, codebook, and rails
    for t in range(4):
        # Rails check
        for r in range(8):
            assert (t, r) in driver.bridge.bram_rails[0], f"Tile {t} rail {r} missing"
        # Codebook check: at least 1 active codebook entry
        assert any(k[0] == t for k in driver.bridge.bram_cb[0].keys()), f"Tile {t} codebook missing"
        # Routes check: 16 routes per tile
        for k in range(16):
            assert (t, k) in driver.bridge.bram_routes[0], f"Tile {t} route {k} missing"


