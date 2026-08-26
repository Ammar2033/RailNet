"""Artifact serialization: checksummed JSON + route-id maps.

Artifact contains NO dense weights (spec 8 / 15).
Route-id map stores the per-element 16-bit route selectors
(bit-pattern indexed) as a separate .npy - this is the honest
dominant storage component documented in the representation
report (dictionary-only metrics are invalid).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1 << 20), b""
        ):

            h.update(chunk)

    return h.hexdigest()


def build_lossless_artifact(
    tensor_name,
    shape,
    rails_arr,
    max_terms,
    route_table,
    exact_unique,
    unique_values,
    attempts=None,
):
    content = {
        "magic": "RNET",

        "version": 1,

        "dtype": "BF16",

        "tensor": tensor_name,

        "shape": [int(x) for x in shape],

        "rail_count": int(len(rails_arr)),

        "max_terms": int(max_terms),

        "rails": [int(b) for b in rails_arr],

        "routes": {
            str(b): [
                [int(rid), int(sgn)]
                for rid, sgn in route
            ]
            for b, route in sorted(
                route_table.items()
            )
        },

        "validation": {
            "unique_values": int(unique_values),

            "exact_unique": int(exact_unique),

            "exact_ratio": float(
                exact_unique / unique_values
                if unique_values else 0.0
            ),

            "runtime_weight_array": "ABSENT",
        },
    }

    if attempts:

        content["compile_attempts"] = attempts

    canonical = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    content["checksum_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()

    return content


def verify_checksum(path):

    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

    stored = data.pop("checksum_sha256")

    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    ok = (
        hashlib.sha256(canonical).hexdigest()
        == stored
    )

    return ok, data


def save_artifact_atomic(path, content):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:

        json.dump(content, f)

    os.replace(tmp, path)


def save_route_map_atomic(path, raw_bits):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".npy.tmp")

    # File-object save prevents numpy's silent ".npy" append.
    with open(tmp, "wb") as f:

        np.save(
            f,
            np.asarray(raw_bits, dtype=np.uint16),
        )

    final = path.with_suffix(".npy") if (
        path.suffix != ".npy"
    ) else path

    os.replace(tmp, final)

    return str(final)
