"""
Runtime engine — device-agnostic dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RailNetDevice:
    kind: str  # cpu | gpu | pcie | fpga
    name: str = ""

    @classmethod
    def cpu(cls) -> RailNetDevice:
        return cls(kind="cpu", name="cpu")

    @classmethod
    def gpu(cls, idx: int = 0) -> RailNetDevice:
        return cls(kind="gpu", name=f"cuda:{idx}")

    @classmethod
    def pcie(cls, dev: str = "railnet0") -> RailNetDevice:
        return cls(kind="pcie", name=dev)

    @classmethod
    def open(cls, name: str = "railnet0") -> RailNetDevice:
        # auto-detect: prefer pcie if driver present else cpu simulator
        return cls(kind="pcie", name=name)

    def load_model(self, artifact_path: str):
        from .transformer import RailNetModel

        return RailNetModel.load(artifact_path, device=self)

    def __repr__(self):
        return f"RailNetDevice({self.kind}:{self.name})"


class RailNetEngine:
    def __init__(self, device: RailNetDevice | None = None):
        self.device = device or RailNetDevice.cpu()

    def dispatch_linear(self, x, compiled):
        if self.device.kind in ("cpu", "pcie"):
            from .linear import rail_linear_fast

            return rail_linear_fast(x, compiled)
        raise NotImplementedError(f"device {self.device.kind} not yet implemented")
