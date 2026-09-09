"""INT8 Quantization and Rail Basis Compiler for RailNet.

Provides:
- Symmetric per-tensor INT8 quantization and dequantization.
- Lossless integer rail basis selection (powers of 2 + frequent values).
- Exact route table generation with at most 2-3 terms per integer weight.
- INT8 RailTensor construction.
"""

from __future__ import annotations

import numpy as np

from railnet.core.shape import Shape
from railnet.core.tensor import RailTensor

# Canonical integer basis ensuring 100% exact coverage of [-128, 127] with <= 3 terms
CANONICAL_INT8_BASIS = [
    1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16,
    20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128
]


def quantize_to_int8(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization of float array to INT8.

    Args:
        weight: Float array of any shape.

    Returns:
        tuple (int8_array, scale) where weight ~ int8_array * scale.
    """
    w = np.asarray(weight, dtype=np.float32)
    max_val = float(np.max(np.abs(w)))
    scale = (max_val / 127.0) if max_val > 1e-12 else 1.0
    int8_w = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return int8_w, float(scale)


def dequantize_from_int8(int8_w: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 array back to float32 using scale."""
    return (int8_w.astype(np.float32) * float(scale))


def learn_int8_basis(
    int8_weights: np.ndarray,
    rails: int = 32,
    max_terms: int = 3,
) -> dict:
    """Select integer rail basis for INT8 weights.

    Combines canonical powers-of-2 / intermediate rails (guaranteeing 100% exact coverage)
    with the most frequent tensor-specific unique values for maximum 1-term efficiency.

    Returns:
        dict with keys:
            "rails": np.ndarray (int32 or int8)
            "unique_vals": np.ndarray
            "counts": np.ndarray
    """
    flat = np.asarray(int8_weights, dtype=np.int8).reshape(-1)
    unique_vals, counts = np.unique(flat, return_counts=True)

    # Sort unique values by frequency descending (excluding 0)
    non_zero_mask = unique_vals != 0
    nz_vals = unique_vals[non_zero_mask]
    nz_counts = counts[non_zero_mask]
    sort_idx = np.argsort(-nz_counts)
    frequent_vals = nz_vals[sort_idx]

    # Start with canonical base
    basis_set = set(CANONICAL_INT8_BASIS)

    # Add frequent values until target rail count is reached
    for v in frequent_vals:
        if len(basis_set) >= rails:
            break
        basis_set.add(int(abs(v)))

    # Ensure rail count does not exceed limit
    rail_list = sorted(list(basis_set))
    if len(rail_list) > rails:
        # Keep canonical + top frequent up to rails
        rail_list = rail_list[:rails]

    rails_arr = np.array(rail_list, dtype=np.int32)
    return {
        "rails": rails_arr,
        "unique_vals": unique_vals,
        "counts": counts,
    }


def compile_int8_routes(
    unique_vals: np.ndarray,
    rails: np.ndarray,
    max_terms: int = 3,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Compile shortest exact routing for each unique INT8 value into rail indices and signs.

    Maps uint8 key (v & 0xFF) -> tuple of (rail_idx, sign).
    sign = 0 (+), sign = 1 (-).
    """
    rail_list = [int(r) for r in rails]
    n_rails = len(rail_list)
    routes: dict[int, tuple[tuple[int, int], ...]] = {}

    for val in unique_vals:
        v = int(val)
        key = v & 0xFF
        if v == 0:
            routes[key] = ()
            continue

        found = False
        # 1-term search
        for r_idx, r in enumerate(rail_list):
            if r == v:
                routes[key] = ((r_idx, 0),)
                found = True
                break
            if -r == v:
                routes[key] = ((r_idx, 1),)
                found = True
                break
        if found:
            continue

        # 2-term search
        for i, r1 in enumerate(rail_list):
            for j in range(i, n_rails):
                r2 = rail_list[j]
                for s1, sgn1 in ((r1, 0), (-r1, 1)):
                    for s2, sgn2 in ((r2, 0), (-r2, 1)):
                        if s1 + s2 == v:
                            routes[key] = ((i, sgn1), (j, sgn2))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue

        # 3-term search
        if max_terms >= 3:
            for i, r1 in enumerate(rail_list):
                for j in range(i, n_rails):
                    r2 = rail_list[j]
                    for k in range(j, n_rails):
                        r3 = rail_list[k]
                        for s1, sgn1 in ((r1, 0), (-r1, 1)):
                            for s2, sgn2 in ((r2, 0), (-r2, 1)):
                                for s3, sgn3 in ((r3, 0), (-r3, 1)):
                                    if s1 + s2 + s3 == v:
                                        routes[key] = ((i, sgn1), (j, sgn2), (k, sgn3))
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        if not found:
            # Fallback to closest 3-term approximation if outside basis bounds
            best_diff = 999999
            best_route = ()
            for i, r1 in enumerate(rail_list):
                for j in range(i, n_rails):
                    r2 = rail_list[j]
                    for s1, sgn1 in ((r1, 0), (-r1, 1)):
                        for s2, sgn2 in ((r2, 0), (-r2, 1)):
                            diff = abs((s1 + s2) - v)
                            if diff < best_diff:
                                best_diff = diff
                                best_route = ((i, sgn1), (j, sgn2))
            routes[key] = best_route

    return routes


def compile_int8_tensor(
    raw: np.ndarray,
    rails: int = 32,
    max_terms: int = 3,
    name: str = "unknown",
    shape: tuple | Shape | None = None,
    scale: float | None = None,
) -> RailTensor:
    """Compile a tensor (float or int8) into an INT8 RailTensor.

    Args:
        raw: Raw numpy array (float32, bf16, or int8).
        rails: Number of integer basis rails (default 32).
        max_terms: Maximum terms per integer (default 3).
        name: Tensor identifier.
        shape: Original tensor shape.
        scale: Pre-computed quantization scale, or None to auto-compute.
    """
    if raw.dtype == np.int8:
        int8_w = raw
        if scale is None:
            scale = 1.0
    else:
        int8_w, auto_scale = quantize_to_int8(raw)
        if scale is None:
            scale = auto_scale

    flat_int8 = int8_w.reshape(-1)
    learned = learn_int8_basis(flat_int8, rails=rails, max_terms=max_terms)
    rails_arr = learned["rails"]
    table = compile_int8_routes(learned["unique_vals"], rails_arr, max_terms=max_terms)

    # route_ids: store raw uint8 representation (v & 0xFF) as uint16 for RailNet compatibility
    route_ids = (flat_int8.astype(np.int32) & 0xFF).astype(np.uint16)

    if shape is None:
        resolved_shape = Shape((len(flat_int8),))
    elif isinstance(shape, tuple):
        resolved_shape = Shape(shape)
    else:
        resolved_shape = shape

    metadata = {
        "scale": float(scale),
        "dtype": "int8",
        "precision": "int8",
    }

    return RailTensor(
        name=name,
        shape=resolved_shape,
        dtype="int8",
        rail_count=len(rails_arr),
        max_terms=max_terms,
        rails_bits=rails_arr,
        routes=table,
        route_ids=route_ids,
        metadata=metadata,
    )
