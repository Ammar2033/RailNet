"""Artifact reader — ``.rnmodel`` header parsing + checksum verification.

Running a model is done from the compiled directory (``railnet.runtime.RailNetModel``);
the single-file packing format is still a design item (docs/ARTIFACT.md)."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from .format import MAGIC


def read_rnmodel_header(path: str) -> dict:
    """Parse a ``.rnmodel`` file's header (magic + version + JSON), verifying
    the checksum. Does not build a runnable model."""
    p = Path(path)
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        struct.unpack("<I", f.read(4))[0]
        hlen = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(hlen).decode())
    stored = header.pop("checksum_sha256", None)
    if stored:
        canon = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canon).hexdigest() != stored:
            raise ValueError("checksum mismatch")
        header["checksum_sha256"] = stored
    return header


def read_rnmodel(path: str, device=None):
    # The single-file .rnmodel packing format does not yet carry everything a
    # RailNetModel needs (config, constants, and the norm/embedding source). Use
    # the compiled directory produced by `railnet compile` / compile_model.
    raise NotImplementedError(
        "loading a runnable model from a single .rnmodel file is not implemented yet "
        "(see docs/ARTIFACT.md) — load the compiled/ directory instead"
    )


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
    except Exception as e:  # noqa: BLE001 - verify must never raise, only report
        return False, {"error": str(e)}
