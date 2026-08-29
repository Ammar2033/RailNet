import base64
import zlib

import numpy as np


def compress_route_ids(route_ids: np.ndarray, method: str = "zlib") -> str:
    """
    Compresses the dense uint16 route_ids array into a base64 encoded string.
    """
    if route_ids.dtype != np.uint16:
        route_ids = route_ids.astype(np.uint16)

    raw_bytes = route_ids.tobytes()

    if method == "zlib":
        compressed = zlib.compress(raw_bytes, level=9)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    return base64.b64encode(compressed).decode("ascii")


def decompress_route_ids(encoded_str: str, method: str = "zlib") -> np.ndarray:
    """
    Decompresses the base64 encoded string back into a flat uint16 route_ids array.
    """
    compressed = base64.b64decode(encoded_str)

    if method == "zlib":
        raw_bytes = zlib.decompress(compressed)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    return np.frombuffer(raw_bytes, dtype=np.uint16)
