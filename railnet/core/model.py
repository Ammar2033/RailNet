from __future__ import annotations

from dataclasses import dataclass, field

from .tensor import RailTensor


@dataclass
class RailModel:
    name: str
    architecture: str
    dtype: str
    tensors: dict[str, RailTensor] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_tensor(self, t: RailTensor):
        self.tensors[t.name] = t

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def to_manifest(self) -> dict:
        return {
            "model": self.name,
            "architecture": self.architecture,
            "dtype": self.dtype,
            "tensors": [t.to_dict() for t in self.tensors.values()],
            "metadata": self.metadata,
        }
