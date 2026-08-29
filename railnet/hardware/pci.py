"""PCIe endpoint model — software shim over CPU simulator."""

from __future__ import annotations

from dataclasses import dataclass

from railnet.runtime.engine import RailNetDevice


@dataclass
class PCIeLink:
    gen: str = "Gen4"
    lanes: int = 16
    bandwidth_gbps: float = 32.0


class PCIeDevice:
    def __init__(self, name: str = "railnet0", link: PCIeLink | None = None):
        self.name = name
        self.link = link or PCIeLink()
        self.device = RailNetDevice.pcie(name)

    def dma_write(self, data: bytes) -> int:
        # shim — copies to host buffer
        return len(data)

    def dma_read(self, nbytes: int) -> bytes:
        return b"\x00" * nbytes

    def load_model(self, artifact: str):
        return self.device.load_model(artifact)
