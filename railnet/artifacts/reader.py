"""Artifact reader — loads .rnmodel or legacy compiled/ directory."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from .format import MAGIC


def read_rnmodel(path: str, device=None):
    p = Path(path)
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        hlen = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(hlen).decode())
    # verify checksum
    stored = header.pop("checksum_sha256", None)
    if stored:
        canon = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canon).hexdigest() != stored:
            raise ValueError("checksum mismatch")
        header["checksum_sha256"] = stored
    # decode route maps
    route_maps = {}
    from railnet.artifacts.compression import decompress_route_ids
    for k, b64 in header.get("route_maps_b64", {}).items():
        route_maps[k] = decompress_route_ids(b64, method="zlib")
    # return a lightweight model handle
    from railnet.runtime.transformer import RailNetModel
    tmp_dir = Path(str(p) + ".unpacked")
    return RailNetModel(header, tmp_dir, device=device)


def verify_rnmodel(path: str) -> tuple[bool, dict]:
    try:
        p = Path(path)
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                return False, {"error": "bad magic"}
            f.read(4)
            hlen = struct.unpack("<I", f.read(4))[0]
            header = json.loads(f.read(hlen).decode())
        stored = header.pop("checksum_sha256", "")
        canon = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        ok = hashlib.sha256(canon).hexdigest() == stored
        header["checksum_sha256"] = stored
        return ok, header
    except Exception as e:
        return False, {"error": str(e)}
