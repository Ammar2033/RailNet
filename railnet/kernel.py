"""RailNet runtime linear kernel (bit-pattern indexed topology).

API contract (Stage-12 rule):

    output = rail_linear(x, compiled)

`compiled` carries rails + topology + this tensor's route-id
map. It NEVER contains or accepts dense weights.
"""

import hashlib
import json
import time

import numpy as np

from railnet.dtypes.bf16 import bf16_array_to_float32


class CompiledTensor:
    def __init__(self, artifact_path, raw_bits=None, shape=None):
        t0 = time.perf_counter()
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stored = data.pop("checksum_sha256")
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.checksum_ok = hashlib.sha256(canonical).hexdigest() == stored
        if not self.checksum_ok:
            raise RuntimeError(f"Artifact checksum FAIL: {artifact_path}")
        self.tensor_name = data.get("tensor", data.get("scope", "<global>"))
        self.rail_count = data["rail_count"]
        self.max_terms = data["max_terms"]
        self.rails_f64 = bf16_array_to_float32(np.array(data["rails"], dtype=np.uint16)).astype(
            np.float64
        )

        rows = 65_536
        mt = self.max_terms
        self.term_rail = np.zeros((rows, mt), dtype=np.int32)
        self.term_sign = np.zeros((rows, mt), dtype=np.int8)
        self.term_active = np.zeros((rows, mt), dtype=bool)

        for bits_str, terms in data["routes"].items():
            g = int(bits_str)
            for t_i, (rid, sgn) in enumerate(terms):
                self.term_rail[g, t_i] = rid
                self.term_sign[g, t_i] = sgn
                self.term_active[g, t_i] = True

        if shape is None:
            shape = data.get("shape")
            if shape is None:
                raise ValueError("shape must be provided or present in the artifact JSON")

        if "compressed_routes" in data:
            from railnet.artifacts.compression import decompress_route_ids

            self.route_ids = (
                decompress_route_ids(data["compressed_routes"]).astype(np.int32).reshape(shape)
            )
        else:
            if raw_bits is None:
                raise ValueError(
                    "raw_bits must be provided if compressed_routes is missing from artifact"
                )
            self.route_ids = raw_bits.astype(np.int32).reshape(shape)

        self.shape = tuple(shape)
        self.out_features = int(shape[0])
        self.in_features = int(shape[1])
        self.load_seconds = time.perf_counter() - t0


def rail_linear(x, compiled):
    c = compiled
    g = c.route_ids.reshape(-1)
    n = g.size
    ii = np.arange(n) % c.in_features
    xv = x[ii]
    jjR = ((np.arange(n) // c.in_features) * c.rail_count).astype(np.int64)

    acc_index = []
    acc_weight = []

    for t in range(c.max_terms):
        act = c.term_active[g, t]
        if not np.any(act):
            continue
        rr = c.term_rail[g[act], t]
        ss = c.term_sign[g[act], t]
        acc_index.append(jjR[act] + rr)
        acc_weight.append(ss * xv[act])

    if acc_index:
        idx = np.concatenate(acc_index)
        wgt = np.concatenate(acc_weight)
        G = np.bincount(idx, weights=wgt, minlength=c.out_features * c.rail_count)
    else:
        G = np.zeros(c.out_features * c.rail_count)

    Y = (G.reshape(c.out_features, c.rail_count) * c.rails_f64[None, :]).sum(axis=1)
    return Y


def prepare(c):
    """
    Hoist invariant index/sign structures out of the hot loop.

    Pure restructuring: the resulting bincount accumulation
    ORDER is identical to rail_linear, so float64 results are
    bit-identical. Called once per tensor before generation.
    """
    if getattr(c, "prepared", False):
        return

    g = c.route_ids.reshape(-1)

    n = g.size

    ii = (np.arange(n) % c.in_features).astype(np.int32)

    jjR = ((np.arange(n) // c.in_features) * c.rail_count).astype(np.int64)

    ii_parts = []

    idx_parts = []

    ss_parts = []

    for t in range(c.max_terms):
        act = c.term_active[g, t]

        if not np.any(act):
            continue

        sel = np.flatnonzero(act)

        ii_parts.append(ii[sel])

        idx_parts.append(jjR[sel] + c.term_rail[g[sel], t])

        ss_parts.append(c.term_sign[g[sel], t])

    if ii_parts:
        c.p_ii = np.concatenate(ii_parts)

        c.p_idx = np.concatenate(idx_parts).astype(np.int64)

        c.p_ss = np.concatenate(ss_parts)

    else:
        c.p_ii = np.zeros(0, dtype=np.int32)

        c.p_idx = np.zeros(0, dtype=np.int64)

        c.p_ss = np.zeros(0, dtype=np.int8)

    c.prepared = True


def rail_linear_fast(x, c):
    """
    Prepared fast path. Bit-identical results to rail_linear
    (same accumulation order), roughly 2x faster per call.
    """

    if not getattr(c, "prepared", False):
        prepare(c)

    xv = x[c.p_ii]

    w = c.p_ss * xv

    G = np.bincount(
        c.p_idx,
        weights=w,
        minlength=c.out_features * c.rail_count,
    )

    return (G.reshape(c.out_features, c.rail_count) * c.rails_f64[None, :]).sum(axis=1)
