"""Artifact writer — produces .rnmodel + file-per-tensor route maps."""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

from .format import MAGIC, VERSION


def write_rnmodel(out_path: str, model_name: str, dtype: str, tensors: list[dict], route_maps: dict[str, np.ndarray] | None = None):
    """
    Write single-file .rnmodel (header + tensor records + inline route maps).
    For legacy directory artifact (compiled/), use save_artifact_atomic.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "magic": MAGIC.decode(),
        "version": VERSION,
        "model": model_name,
        "dtype": dtype,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    # embed route_ids as base64 if provided (small tensors)
    if route_maps:
        from railnet.artifacts.compression import compress_route_ids
        embedded = {}
        for k, arr in route_maps.items():
            embedded[k] = compress_route_ids(arr, method="zlib")
        manifest["route_maps_b64"] = embedded

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["checksum_sha256"] = hashlib.sha256(canonical).hexdigest()

    # simple framing: [u32 header_len][json header][optional blobs]
    header_json = json.dumps(manifest).encode()
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
    return str(out_path)


def write_manifest_json(out_path: str, content: dict):
    import json as _json
    from pathlib import Path as _P
    p = _P(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(content, f, indent=2)
    os.replace(tmp, p)
