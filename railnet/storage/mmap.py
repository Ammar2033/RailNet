"""Embedding mmap — exact row lookup, zero-copy."""

from __future__ import annotations

import mmap
from pathlib import Path

import numpy as np


class EmbeddingMMap:
    def __init__(self, path: str, shape: tuple[int, int], dtype=np.uint16):
        self.path = Path(path)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._f = None
        self._mm = None
        self._arr = None

    def open(self):
        self._f = open(self.path, "rb")  # noqa: SIM115 - held open for the mmap lifetime
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._arr = np.frombuffer(self._mm, dtype=self.dtype).reshape(self.shape)
        return self

    def lookup(self, token_id: int) -> np.ndarray:
        if self._arr is None:
            self.open()
        assert self._arr is not None
        return self._arr[int(token_id)].copy()

    def close(self):
        self._arr = None
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._f:
            self._f.close()
            self._f = None
