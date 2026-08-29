"""Embedding strategy for Stage 15: exact mmap row lookup.

The embedding matrix is NOT loaded densely and NOT compressed. It is mmap'd
from the safetensors file as a ``(vocab, hidden)`` uint16 view; rows are read
by fancy-indexing and the OS pages in only what is touched. The tied LM head
streams the same view in row chunks (never a full float materialization).
Compression: NOT CLAIMED (spec 50).
"""

import mmap

import numpy as np

from . import safetensors_reader as SR


def _bits_to_f64(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class MmapRowLookup:
    def __init__(self, tensor_name="model.embed_tokens.weight", model_file=None):
        model_file = model_file or SR.MODEL_FILE
        meta = SR.tensor_metadata(tensor_name, model_file=model_file)
        assert meta["dtype"] == "BF16"

        self.name = tensor_name
        self.vocab, self.hidden = meta["shape"]
        self.model_file = model_file

        self._f = open(model_file, "rb")  # noqa: SIM115 - held open for the mmap lifetime
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        end = meta["offset"] + self.vocab * self.hidden * 2
        self._rows = np.frombuffer(self._mm[meta["offset"] : end], dtype=np.uint16).reshape(
            self.vocab, self.hidden
        )

    def close(self):
        self._rows = None
        self._mm.close()
        self._f.close()

    def rows_f64(self, token_ids) -> np.ndarray:
        """Exact row lookup; returns ``(len(token_ids), hidden)`` float64."""
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab):
            raise IndexError(f"token id out of vocab [0, {self.vocab})")
        return _bits_to_f64(self._rows[ids])

    def logits_chunked(self, h_last, chunk_rows=32768) -> np.ndarray:
        """Tied LM head: ``logits = E @ h`` in row chunks so the embedding is
        never fully materialized as float. Returns ``(vocab,)`` float64."""
        h = np.asarray(h_last, dtype=np.float64)
        out = np.empty(self.vocab, dtype=np.float64)
        for start in range(0, self.vocab, chunk_rows):
            stop = min(start + chunk_rows, self.vocab)
            out[start:stop] = _bits_to_f64(self._rows[start:stop]) @ h
        return out
