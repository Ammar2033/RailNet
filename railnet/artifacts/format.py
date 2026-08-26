"""
Artifact format — .rnmodel container.

Header: magic RNET, version, dtype, model metadata, tensor manifest, checksums.
Tensors stored as:
  - rail table (uint16 bits + topology dict)
  - route_ids (separate .npy / embedded blob)

Spec guarantee: artifact MUST NOT contain dense weight array.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

MAGIC = b"RNET"
VERSION = 1


@dataclass
class ArtifactHeader:
    magic: bytes = MAGIC
    version: int = VERSION
    model: str = "gemma3"
    dtype: str = "bf16"
    tensor_count: int = 0


@dataclass
class TensorRecord:
    name: str
    shape: list[int]
    dtype: str
    rail_count: int
    max_terms: int
    rails_bits: list[int]
    routes: dict  # str(bits) -> [[rail,sign],...]
    checksum: str = ""

    def canonical_bytes(self) -> bytes:
        canon = json.dumps({
            "name": self.name, "shape": self.shape, "dtype": self.dtype,
            "rail_count": self.rail_count, "max_terms": self.max_terms,
            "rails": self.rails_bits, "routes": self.routes,
        }, sort_keys=True, separators=(",", ":")).encode()
        return canon

    def compute_checksum(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
