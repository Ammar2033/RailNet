"""Manifest helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def build_manifest(model: str, dtype: str, tensors: list[dict]) -> dict:
    return {
        "magic": "RNET",
        "version": 1,
        "model": model,
        "dtype": dtype,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "runtime_weight_array": "ABSENT",
    }


def checksum_manifest(manifest: dict) -> str:
    canon = json.dumps({k: v for k, v in manifest.items() if k != "checksum_sha256"}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canon).hexdigest()


def save_manifest(path: str, manifest: dict):
    import os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["checksum_sha256"] = checksum_manifest(manifest)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, p)
