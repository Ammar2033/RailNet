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
        from .pcie import RailNetPCIeDriver

        device = cls(kind="pcie", name=dev)
        device.pcie_driver = RailNetPCIeDriver(device_name=dev)
        return device

    @classmethod
    def open(cls, name: str = "railnet0") -> RailNetDevice:
        # auto-detect: prefer pcie if driver present else cpu simulator
        return cls.pcie(dev=name)

    @classmethod
    def sim(
        cls,
        profile: str = "edge_asic",
        tiles: int | None = None,
        freq_mhz: float | None = None,
        pcie_gen: str | None = None,
    ) -> RailNetDevice:
        from railnet.sim.accelerator import AcceleratorSimulator, HardwareConfig

        if profile == "cloud_asic":
            cfg = HardwareConfig.cloud_asic()
        elif profile == "fpga_proto":
            cfg = HardwareConfig.fpga_proto()
        else:
            cfg = HardwareConfig.edge_asic()

        if tiles is not None:
            cfg.tiles = tiles
        if freq_mhz is not None:
            cfg.freq_mhz = freq_mhz
        if pcie_gen is not None:
            cfg.pcie_gen = pcie_gen

        dev = cls(kind="sim", name=f"sim:{cfg.name}")
        dev.sim_engine = AcceleratorSimulator(cfg)
        return dev

    def get_telemetry(self, tokens_processed: int | None = None):
        if hasattr(self, "sim_engine") and self.sim_engine:
            return self.sim_engine.finalize_telemetry(tokens_processed or 1)
        return None

    def reset_telemetry(self):
        if hasattr(self, "sim_engine") and self.sim_engine:
            self.sim_engine.reset_telemetry()

    def close(self):
        if hasattr(self, "pcie_driver") and self.pcie_driver is not None:
            self.pcie_driver.close()

    def load_model(self, artifact_path: str):
        from .transformer import RailNetModel

        return RailNetModel.load(artifact_path, device=self)

    def __repr__(self):
        return f"RailNetDevice({self.kind}:{self.name})"


class RailNetEngine:
    def __init__(self, device: RailNetDevice | None = None):
        self.device = device or RailNetDevice.cpu()

    def dispatch_linear(self, x, compiled):
        if self.device.kind == "pcie":
            driver = getattr(self.device, "pcie_driver", None)
            if driver is None:
                from .pcie import RailNetPCIeDriver

                driver = RailNetPCIeDriver(self.device.name)
                self.device.pcie_driver = driver
            return driver.dispatch_linear(x, compiled)
        elif self.device.kind == "cpu":
            from .linear import rail_linear_fast

            return rail_linear_fast(x, compiled)
        raise NotImplementedError(f"device {self.device.kind} not yet implemented")
