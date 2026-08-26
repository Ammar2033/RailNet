"""Memory accounting — static vs dynamic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryBudget:
    static_rails_bytes: int = 0
    static_topology_bytes: int = 0
    dynamic_activation_bytes: int = 0
    dynamic_kv_bytes: int = 0

    @property
    def static_total(self) -> int:
        return self.static_rails_bytes + self.static_topology_bytes

    @property
    def dynamic_total(self) -> int:
        return self.dynamic_activation_bytes + self.dynamic_kv_bytes

    @property
    def total(self) -> int:
        return self.static_total + self.dynamic_total

    def to_dict(self) -> dict:
        return {
            "static_rails_bytes": self.static_rails_bytes,
            "static_topology_bytes": self.static_topology_bytes,
            "static_total": self.static_total,
            "dynamic_activation_bytes": self.dynamic_activation_bytes,
            "dynamic_kv_bytes": self.dynamic_kv_bytes,
            "dynamic_total": self.dynamic_total,
            "total": self.total,
        }
