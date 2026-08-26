"""
Hardware architecture — PCIe card model (PLANNED / RESEARCH).

Not yet silicon; documents target organization.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RailFabric:
    rail_count: int = 96
    dtype: str = "bf16"
    programmable: bool = True  # values loaded from artifact


@dataclass
class RoutingFabric:
    max_terms: int = 4
    storage: str = "SRAM"  # research: SRAM / registers / ROM-like
    compressed: bool = False


@dataclass
class ComputeFabric:
    shared_multipliers: int = 653_800_000  # upper bound full model
    reduction_network: str = "adder-tree"


@dataclass
class MemorySubsystem:
    activation_buf_kb: int = 1024
    kv_cache: str = "HBM / LPDDR (research)"


@dataclass
class RailNetCard:
    rail_fabric: RailFabric = field(default_factory=RailFabric)
    routing: RoutingFabric = field(default_factory=RoutingFabric)
    compute: ComputeFabric = field(default_factory=ComputeFabric)
    memory: MemorySubsystem = field(default_factory=MemorySubsystem)
    pcie_gen: str = "Gen4 x16"

    def to_dict(self) -> dict:
        return {
            "rail_fabric": self.rail_fabric.__dict__,
            "routing": self.routing.__dict__,
            "compute": self.compute.__dict__,
            "memory": self.memory.__dict__,
            "pcie": self.pcie_gen,
            "status": "RESEARCH / NOT YET SILICON",
        }
