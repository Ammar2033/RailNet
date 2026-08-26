"""Embedding strategy for Stage 15: exact mmap row lookup.

The embedding matrix is NOT loaded densely and NOT compressed.
Token rows are read lazily from the safetensors file via mmap
(exact lookup artifact). Compression: NOT CLAIMED (spec 50).
"""
import numpy as np

from . import safetensors_reader as SR


class MmapRowLookup:

    def __init__(
        self,
        tensor_name="model.embed_tokens.weight",
    ):
        meta = SR.tensor_metadata(tensor_name)

        assert meta["dtype"] == "BF16"

        self.name = tensor_name

        self.vocab, self.hidden = meta["shape"]

        self.offset = meta["offset"]

        self.row_bytes = self.hidden * 2

        self.model_file = SR.MODEL_FILE

    def rows_f64(self, token_ids):
        """
        Exact row lookup; only requested rows are touched.

        Returns (len(token_ids), hidden) float64.
        """
        ids = np.asarray(token_ids, dtype=np.int64)

        out = np.empty(
            (ids.size, self.hidden), dtype=np.float64
        )

        with open(self.model_file, "rb") as f:

            for i, tid in enumerate(ids.tolist()):

                if not (0 <= tid < self.vocab):

                    raise IndexError(
                        f"token id {tid} out of vocab"
                    )

                f.seek(
                    self.offset + tid * self.row_bytes
                )

                buf = f.read(self.row_bytes)

                bits = np.frombuffer(
                    buf, dtype=np.uint16
                )

                fp32 = (
                    bits.astype(np.uint32) << 16
                ).view(np.float32)

                out[i] = fp32.astype(np.float64)

        return out

    # ---- LM head (tied weights) --------------------------

    def logits_chunked(
        self,
        h_last,
        chunk_rows=16384,
    ):
        """
        Tied LM head: logits = h @ E^T computed in chunks so
        the full embedding never materializes in RAM.

        Returns (vocab,) float64.
        """

        h = np.asarray(h_last, dtype=np.float64)

        out = np.empty(self.vocab, dtype=np.float64)

        with open(self.model_file, "rb") as f:

            for start in range(
                0, self.vocab, chunk_rows
            ):

                stop = min(
                    start + chunk_rows, self.vocab
                )

                n = stop - start

                block = np.empty(
                    (n, self.hidden), dtype=np.float64
                )

                for r in range(n):

                    f.seek(
                        self.offset
                        + (start + r) * self.row_bytes
                    )

                    bits = np.frombuffer(
                        f.read(self.row_bytes),
                        dtype=np.uint16,
                    )

                    fp32 = (
                        bits.astype(np.uint32) << 16
                    ).view(np.float32)

                    block[r] = fp32.astype(np.float64)

                out[start:stop] = block @ h

        return out
