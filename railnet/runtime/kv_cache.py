"""
KV cache — per-layer growing cache.

Exact semantics verified in 14_gemma_multi_block.py
"""
from __future__ import annotations

import numpy as np


class KVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.caches: list[dict | None] = [None] * num_layers

    def get(self, layer_idx: int):
        return self.caches[layer_idx]

    def update(self, layer_idx: int, cache: dict):
        self.caches[layer_idx] = cache

    def clear(self):
        self.caches = [None] * self.num_layers

    def seq_len(self, layer_idx: int) -> int:
        c = self.caches[layer_idx]
        if c is None:
            return 0
        return c["K"].shape[1]
