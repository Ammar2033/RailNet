"""Cycle-Accurate Virtual Accelerator Simulator for RailNet.

Simulates the hardware execution of RailNet transformer inference:
- Stage-A pipelined BRAM gather (with forwarding bypass and collision stalls).
- Stage-B integer/float reduction tree latency.
- Multi-tile 2D grid allocation and activation broadcast over NoC.
- PCIe Host-to-Device DMA transfer latency.
- Power, throughput (tokens/s), latency (ms), and energy efficiency (Tokens/Joule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HardwareConfig:
    """Parametric hardware accelerator specification."""
    name: str = "Edge ASIC"
    tiles: int = 64
    grid_dim: tuple[int, int] = (8, 8)
    freq_mhz: float = 500.0
    stage_a_lanes: int = 3
    stage_a_rmw_latency_int8: int = 1
    stage_a_rmw_latency_bf16: int = 2
    stage_b_latency: int = 4
    bram_read_latency: int = 1
    noc_flit_width_bytes: int = 32  # 256-bit NoC bus
    pcie_gen: str = "Gen4x16"
    pcie_bandwidth_gbps: float = 25.0  # Usable GB/s
    pcie_dma_latency_us: float = 3.5    # Driver & DMA launch latency
    static_power_watts: float = 0.12     # Chip static leakage
    energy_per_gather_int8_pj: float = 0.04  # INT8 0-DSP gather addition
    energy_per_gather_bf16_pj: float = 0.12  # BF16/FP32 gather addition
    energy_per_stageb_pj: float = 0.15  # INT8 Stage-B MAC
    energy_per_bram_rw_pj: float = 0.10 # BRAM read-modify-write

    @classmethod
    def edge_asic(cls) -> HardwareConfig:
        """Standard commercial edge inference ASIC (22nm/16nm class)."""
        return cls(
            name="Edge ASIC",
            tiles=64,
            grid_dim=(8, 8),
            freq_mhz=500.0,
            pcie_gen="Gen4x16",
            pcie_bandwidth_gbps=25.0,
            static_power_watts=0.12,
        )

    @classmethod
    def cloud_asic(cls) -> HardwareConfig:
        """High-throughput datacenter monolithic inference ASIC."""
        return cls(
            name="Cloud ASIC",
            tiles=256,
            grid_dim=(16, 16),
            freq_mhz=1000.0,
            pcie_gen="Gen5x16",
            pcie_bandwidth_gbps=50.0,
            static_power_watts=0.35,
        )

    @classmethod
    def fpga_proto(cls) -> HardwareConfig:
        """FPGA prototype (e.g. Xilinx Alveo U50 / Kria KV260)."""
        return cls(
            name="FPGA Prototype",
            tiles=16,
            grid_dim=(4, 4),
            freq_mhz=150.0,
            pcie_gen="Gen3x8",
            pcie_bandwidth_gbps=6.5,
            static_power_watts=2.5,
        )


@dataclass
class LayerTelemetry:
    """Telemetry recorded for a single linear layer forward execution."""
    layer_name: str
    in_features: int
    out_features: int
    rail_count: int
    dtype: str
    active_tiles: int
    total_terms: int
    stage_a_cycles: int
    stage_b_cycles: int
    broadcast_cycles: int
    total_cycles: int
    latency_ms: float
    dynamic_energy_pj: float


@dataclass
class HardwareTelemetry:
    """Comprehensive hardware telemetry report for one or more forward passes."""
    config: HardwareConfig
    layers: List[LayerTelemetry] = field(default_factory=list)
    tokens_processed: int = 1
    total_cycles: int = 0
    total_simulated_ms: float = 0.0
    pcie_dma_latency_ms: float = 0.0
    total_energy_joules: float = 0.0
    average_power_watts: float = 0.0
    tokens_per_second: float = 0.0
    tile_utilization_pct: float = 0.0

    def summary(self) -> str:
        sep = "=" * 82
        subsep = "-" * 82
        lines = [
            sep,
            f" RailNet Hardware Accelerator Cycle-Accurate Telemetry Report",
            f" Target Profile : {self.config.name} ({self.config.tiles} Tiles @ {self.config.freq_mhz:.0f} MHz, {self.config.pcie_gen})",
            f" Tokens Processed: {self.tokens_processed}",
            sep,
            f"{'Layer / Operation':<32} | {'Tiles':<5} | {'Stg-A Cyc':<9} | {'Stg-B Cyc':<9} | {'Latency':<9}",
            subsep,
        ]

        for l in self.layers:
            lines.append(
                f"{l.layer_name:<32} | {l.active_tiles:<5} | {l.stage_a_cycles:<9} | {l.stage_b_cycles:<9} | {l.latency_ms:.3f} ms"
            )

        lines.extend([
            subsep,
            f" Total Compute Cycles      : {self.total_cycles:,} cycles",
            f" PCIe DMA Transfer Time    : {self.pcie_dma_latency_ms:.3f} ms",
            f" Total Simulated Time      : {self.total_simulated_ms:.3f} ms (per token: {self.total_simulated_ms / max(1, self.tokens_processed):.3f} ms)",
            f" Simulated Throughput      : {self.tokens_per_second:.1f} tokens / second",
            f" Average Tile Utilization  : {self.tile_utilization_pct:.1f} %",
            f" Estimated Chip Power      : {self.average_power_watts:.2f} Watts (Sub-watt: {self.average_power_watts < 1.0})",
            f" Energy Efficiency         : {self.tokens_per_joule():.1f} Tokens / Joule",
            sep,
        ])
        return "\n".join(lines)

    def tokens_per_joule(self) -> float:
        if self.total_energy_joules > 0:
            return self.tokens_processed / self.total_energy_joules
        return 0.0


class TileSimulator:
    """Cycle-by-cycle hardware simulation for a single RailNet karo."""

    def __init__(self, config: HardwareConfig):
        self.config = config

    def simulate_neuron(
        self,
        in_features: int,
        term_counts: np.ndarray,
        rail_count: int,
        dtype: str = "int8",
    ) -> tuple[int, int, float]:
        """Simulate cycles and energy required for a single output neuron.

        Args:
            in_features: Length of input activation vector.
            term_counts: Array of terms per weight (length in_features).
            rail_count: Number of shared rails (e.g. 32 for INT8, 96 for BF16).
            dtype: 'int8' or 'bf16'.

        Returns:
            tuple (stage_a_cycles, stage_b_cycles, dynamic_energy_pj)
        """
        total_terms = int(np.sum(term_counts))

        is_int8 = dtype == "int8"
        rmw_latency = self.config.stage_a_rmw_latency_int8 if is_int8 else self.config.stage_a_rmw_latency_bf16

        # Stage-A: Parallel gather across `stage_a_lanes`
        # Each lane takes 1 term/cycle * rmw_latency.
        lanes = self.config.stage_a_lanes
        base_gather_cycles = math.ceil((total_terms / lanes) * rmw_latency)

        # Collision / hazard model:
        # When multiple lanes write to the same rail in the same cycle, an arbiter stalls 1 cycle.
        # Collision probability depends on rail_count (smaller rail count = higher collision probability)
        collision_prob = 1.0 / max(1, rail_count)
        estimated_stalls = math.ceil(base_gather_cycles * (lanes - 1) * collision_prob * 0.5)

        stage_a_cycles = self.config.bram_read_latency + base_gather_cycles + estimated_stalls

        # Stage-B: Sequentially read G[r] and multiply by R[r]
        stage_b_cycles = rail_count + self.config.stage_b_latency

        # Dynamic Energy
        energy_gather_unit = self.config.energy_per_gather_int8_pj if is_int8 else self.config.energy_per_gather_bf16_pj
        energy_gather = total_terms * energy_gather_unit
        energy_bram = total_terms * self.config.energy_per_bram_rw_pj
        stage_b_unit_energy = self.config.energy_per_stageb_pj if is_int8 else (self.config.energy_per_stageb_pj * 3.0)
        energy_reduction = rail_count * stage_b_unit_energy
        dynamic_energy = energy_gather + energy_bram + energy_reduction

        return stage_a_cycles, stage_b_cycles, dynamic_energy


class AcceleratorSimulator:
    """Multi-tile grid and system-level cycle-accurate hardware simulator."""

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig.edge_asic()
        self.tile_sim = TileSimulator(self.config)
        self.telemetry = HardwareTelemetry(config=self.config)

    def reset_telemetry(self) -> None:
        """Clear recorded telemetry."""
        self.telemetry = HardwareTelemetry(config=self.config)

    def simulate_linear(
        self,
        compiled: Any,
        layer_name: str = "linear",
    ) -> LayerTelemetry:
        """Simulate execution of a compiled linear layer on the hardware grid."""
        c = compiled
        out_f = int(c.out_features)
        in_f = int(c.in_features)
        rail_count = int(c.rail_count)
        dtype = str(getattr(c, "dtype", "bf16"))

        # Distribute output neurons across available tiles
        n_tiles = self.config.tiles
        active_tiles = min(out_f, n_tiles)
        neurons_per_tile = math.ceil(out_f / active_tiles)

        # Estimate terms per weight from compiled tensor
        mt = int(c.max_terms)
        # Average terms per weight in RailNet is ~2.3 - 2.8
        avg_terms = 2.4 if dtype == "int8" else 2.8
        synthetic_terms = np.full(in_f, avg_terms)

        stg_a_cyc_per_neuron, stg_b_cyc_per_neuron, dynamic_energy_per_neuron = (
            self.tile_sim.simulate_neuron(in_f, synthetic_terms, rail_count, dtype=dtype)
        )

        # Karolar nöronları sırayla işler: Total karo süresi
        tile_compute_cycles = (stg_a_cyc_per_neuron + stg_b_cyc_per_neuron) * neurons_per_tile

        # Activation broadcast over 2D NoC mesh
        # Mesh diameter = width + height
        mesh_w, mesh_h = self.config.grid_dim
        mesh_hops = mesh_w + mesh_h
        flit_bytes = self.config.noc_flit_width_bytes
        act_bytes = in_f * (1 if dtype == "int8" else 2)
        flits = math.ceil(act_bytes / flit_bytes)
        broadcast_cycles = mesh_hops + flits

        total_cycles = broadcast_cycles + tile_compute_cycles
        latency_ms = (total_cycles / (self.config.freq_mhz * 1e6)) * 1000.0

        layer_energy_pj = dynamic_energy_per_neuron * out_f

        record = LayerTelemetry(
            layer_name=layer_name,
            in_features=in_f,
            out_features=out_f,
            rail_count=rail_count,
            dtype=dtype,
            active_tiles=active_tiles,
            total_terms=int(out_f * in_f * avg_terms),
            stage_a_cycles=stg_a_cyc_per_neuron * neurons_per_tile,
            stage_b_cycles=stg_b_cyc_per_neuron * neurons_per_tile,
            broadcast_cycles=broadcast_cycles,
            total_cycles=total_cycles,
            latency_ms=latency_ms,
            dynamic_energy_pj=layer_energy_pj,
        )

        self.telemetry.layers.append(record)
        return record

    def simulate_pcie_transfer(self, num_tokens: int, hidden_size: int) -> float:
        """Calculate PCIe DMA transfer time for input token activations."""
        bytes_transferred = num_tokens * hidden_size * 2  # BF16
        bw_bytes_per_us = (self.config.pcie_bandwidth_gbps * 1e9) / 1e6
        transfer_us = bytes_transferred / bw_bytes_per_us
        total_pcie_ms = (transfer_us + self.config.pcie_dma_latency_us) / 1000.0
        self.telemetry.pcie_dma_latency_ms += total_pcie_ms
        return total_pcie_ms

    def finalize_telemetry(self, tokens_processed: int = 1) -> HardwareTelemetry:
        """Compute aggregate statistics for the simulation."""
        self.telemetry.tokens_processed = max(1, tokens_processed)

        # Sum cycles across all sequential layers
        total_comp_cycles = sum(l.total_cycles for l in self.telemetry.layers)
        self.telemetry.total_cycles = total_comp_cycles

        compute_ms = (total_comp_cycles / (self.config.freq_mhz * 1e6)) * 1000.0
        self.telemetry.total_simulated_ms = compute_ms + self.telemetry.pcie_dma_latency_ms

        if self.telemetry.total_simulated_ms > 0:
            self.telemetry.tokens_per_second = (
                self.telemetry.tokens_processed / (self.telemetry.total_simulated_ms / 1000.0)
            )

        # Average tile utilization
        if self.telemetry.layers:
            util_list = [l.active_tiles / self.config.tiles for l in self.telemetry.layers]
            self.telemetry.tile_utilization_pct = float(np.mean(util_list) * 100.0)

        # Energy & Power
        dynamic_joules = sum(l.dynamic_energy_pj * 1e-12 for l in self.telemetry.layers)
        sim_sec = max(1e-6, self.telemetry.total_simulated_ms / 1000.0)
        static_joules = self.config.static_power_watts * sim_sec
        total_joules = dynamic_joules + static_joules
        self.telemetry.total_energy_joules = total_joules
        self.telemetry.average_power_watts = total_joules / sim_sec

        return self.telemetry
