"""Streaming safetensors access: header metadata + mmap slice reads.

Never materializes the whole model (spec 14).
"""
import importlib.util
import json
import math
import mmap
import struct
from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).resolve().parent.parent / (
    "model_data/model.safetensors"
)

_HEADER_CACHE = None


def read_header(model_file=MODEL_FILE):
    global _HEADER_CACHE
    if _HEADER_CACHE is None or _HEADER_CACHE[2] != str(model_file):
        with open(model_file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        _HEADER_CACHE = (header, 8 + n, str(model_file))
    return _HEADER_CACHE[0], _HEADER_CACHE[1]


def tensor_metadata(name, model_file=MODEL_FILE):
    header, base = read_header(model_file)
    if name not in header:
        raise KeyError(name)
    meta = header[name]
    shape = tuple(int(x) for x in meta["shape"])
    start, end = meta["data_offsets"]
    return {
        "name": name,
        "dtype": meta["dtype"],
        "shape": shape,
        "nbytes": end - start,
        "offset": base + start,
    }


def read_tensor_raw(name, model_file=MODEL_FILE):
    """Read one tensor as uint16 numpy array via mmap slice."""
    meta = tensor_metadata(name, model_file)
    assert meta["dtype"] == "BF16"
    with open(model_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            raw = np.frombuffer(
                mm[meta["offset"]: meta["offset"] + meta["nbytes"]],
                dtype=np.uint16,
            ).copy()
    return raw, meta["shape"]


def list_tensors(model_file=MODEL_FILE):
    header, _ = read_header(model_file)
    return [k for k in header.keys() if k != "__metadata__"]
