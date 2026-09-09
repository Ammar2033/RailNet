"""Unit tests for the RailNet cycle-accurate virtual accelerator simulator."""

import math
from types import SimpleNamespace
import numpy as np
import pytest

from railnet.sim.accelerator import (
    HardwareConfig,
    TileSimulator,
    AcceleratorSimulator,
    HardwareTelemetry,
    LayerTelemetry,
)


def test_hardware_configs():
    edge = HardwareConfig.edge_asic()
    assert edge.tiles == 64
    assert edge.freq_mhz == 500.0
    assert edge.grid_dim == (8, 8)
    assert edge.stage_a_rmw_latency_int8 == 1
    assert edge.stage_a_rmw_latency_bf16 == 2

    cloud = HardwareConfig.cloud_asic()
    assert cloud.tiles == 256
    assert cloud.freq_mhz == 1000.0
    assert cloud.grid_dim == (16, 16)

    fpga = HardwareConfig.fpga_proto()
    assert fpga.tiles == 16
    assert fpga.freq_mhz == 150.0
    assert fpga.grid_dim == (4, 4)


def test_tile_simulator_int8_vs_bf16():
    cfg = HardwareConfig.edge_asic()
    sim = TileSimulator(cfg)

    in_features = 64
    rail_count = 32
    # 2.5 terms per weight on average
    term_counts = np.full(in_features, 2.5)

    stg_a_int8, stg_b_int8, energy_int8 = sim.simulate_neuron(
        in_features, term_counts, rail_count, dtype="int8"
    )
    stg_a_bf16, stg_b_bf16, energy_bf16 = sim.simulate_neuron(
        in_features, term_counts, rail_count, dtype="bf16"
    )

    # INT8 has 1-cycle RMW vs BF16 2-cycle RMW
    assert stg_a_int8 < stg_a_bf16
    # INT8 dynamic energy is significantly lower than BF16
    assert energy_int8 < energy_bf16
    assert energy_int8 > 0.0
    assert stg_b_int8 > 0


def test_tile_simulator_forward_bypass_hazard():
    cfg = HardwareConfig.edge_asic()
    sim = TileSimulator(cfg)

    # In single neuron, consecutive hits on the same rail trigger a 1-cycle stall
    in_features = 128
    rail_count = 16  # low rail count -> more collisions
    term_counts = np.full(in_features, 3.0)

    stg_a, stg_b, _ = sim.simulate_neuron(in_features, term_counts, rail_count, dtype="int8")
    total_terms = int(np.sum(term_counts))
    base_cycles = math.ceil(total_terms / cfg.stage_a_lanes) * cfg.stage_a_rmw_latency_int8
    # cycles must include both RMW latency and collision stalls
    assert stg_a > base_cycles


def test_accelerator_linear_simulation():
    cfg = HardwareConfig.edge_asic()
    acc = AcceleratorSimulator(cfg)

    # Mock compiled tensor
    mock_compiled = SimpleNamespace(
        out_features=256,
        in_features=128,
        rail_count=32,
        max_terms=3,
        dtype="int8",
    )

    layer_tel = acc.simulate_linear(mock_compiled, layer_name="layer_0.q_proj")
    assert layer_tel.layer_name == "layer_0.q_proj"
    assert layer_tel.active_tiles == min(256, cfg.tiles)
    assert layer_tel.total_cycles > 0
    assert layer_tel.latency_ms > 0.0
    assert layer_tel.broadcast_cycles > 0

    # Ensure layer is recorded
    assert len(acc.telemetry.layers) == 1


def test_accelerator_pcie_transfer():
    cfg = HardwareConfig.edge_asic()
    acc = AcceleratorSimulator(cfg)

    latency_ms = acc.simulate_pcie_transfer(num_tokens=4, hidden_size=2048)
    assert latency_ms > 0.0
    assert acc.telemetry.pcie_dma_latency_ms == latency_ms

    # Second transfer accumulates
    acc.simulate_pcie_transfer(num_tokens=1, hidden_size=2048)
    assert acc.telemetry.pcie_dma_latency_ms > latency_ms


def test_finalize_telemetry_and_summary():
    cfg = HardwareConfig.edge_asic()
    acc = AcceleratorSimulator(cfg)

    mock_c1 = SimpleNamespace(out_features=128, in_features=128, rail_count=32, max_terms=3, dtype="bf16")
    mock_c2 = SimpleNamespace(out_features=256, in_features=128, rail_count=32, max_terms=3, dtype="int8")

    acc.simulate_linear(mock_c1, "layer_0.q_proj")
    acc.simulate_linear(mock_c2, "layer_0.gate_proj")
    acc.simulate_pcie_transfer(num_tokens=1, hidden_size=128)

    tel = acc.finalize_telemetry(tokens_processed=2)
    assert tel.total_cycles > 0
    assert tel.total_simulated_ms > 0.0
    assert tel.tokens_per_second > 0.0
    assert tel.average_power_watts > 0.0
    # Edge ASIC design should be sub-watt (< 1.0W)
    assert tel.average_power_watts < 1.0
    assert tel.tokens_per_joule() > 0.0
    assert 0.0 < tel.tile_utilization_pct <= 100.0

    summary_str = tel.summary()
    assert "RailNet Hardware Accelerator Cycle-Accurate Telemetry Report" in summary_str
    assert "layer_0.q_proj" in summary_str
    assert "layer_0.gate_proj" in summary_str
    assert "PCIe DMA Transfer Time" in summary_str
    assert "Sub-watt: True" in summary_str


def test_reset_telemetry():
    cfg = HardwareConfig.edge_asic()
    acc = AcceleratorSimulator(cfg)

    mock_c = SimpleNamespace(out_features=64, in_features=64, rail_count=32, max_terms=3, dtype="int8")
    acc.simulate_linear(mock_c, "test_layer")
    assert len(acc.telemetry.layers) == 1

    acc.reset_telemetry()
    assert len(acc.telemetry.layers) == 0
    assert acc.telemetry.total_cycles == 0
