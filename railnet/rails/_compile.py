import numpy as np
import math
from railnet.dtypes.bf16 import (
    bf16_bits_to_float32, bf16_array_to_float32, 
    float32_to_bf16_bits, fp32_array_to_bf16_bits, bf16_bitwise_equal
)

def greedy_routes(
    target_values,
    rails,
    max_terms
):
    """
    Vectorized sparse routing.

    Each target is represented as:

        target ≈ ±R1 ±R2 ...

    At most max_terms rails are used.
    """

    n = len(
        target_values
    )

    r = len(
        rails
    )

    rail_values = (
        bf16_array_to_float32(
            rails
        ).astype(
            np.float64
        )
    )

    residual = (
        target_values.copy()
    )

    routes = np.zeros(
        (
            n,
            max_terms
        ),
        dtype=np.int16
    )

    signs = np.zeros(
        (
            n,
            max_terms
        ),
        dtype=np.int8
    )

    used = np.zeros(
        (
            n,
            r
        ),
        dtype=bool
    )

    active_terms = np.zeros(
        n,
        dtype=np.int8
    )

    for term in range(
        max_terms
    ):

        pos_error = np.abs(
            residual[:, None]
            -
            rail_values[None, :]
        )

        neg_error = np.abs(
            residual[:, None]
            +
            rail_values[None, :]
        )

        choose_positive = (
            pos_error
            <=
            neg_error
        )

        error = np.minimum(
            pos_error,
            neg_error
        )

        error[
            used
        ] = np.inf

        best_rail = np.argmin(
            error,
            axis=1
        )

        rows = np.arange(
            n
        )

        best_error = (
            error[
                rows,
                best_rail
            ]
        )

        valid = np.isfinite(
            best_error
        )

        if not np.any(
            valid
        ):
            break

        # ----------------------------------------------------
        # Sparse: only add a rail if it strictly reduces
        # the absolute residual. This ensures `max_terms`
        # is an upper bound, not a forced count, and gives
        # monotonic improvement: more terms never worsens.
        # ----------------------------------------------------

        abs_residual = np.abs(
            residual
        )

        improves = (
            best_error
            <
            abs_residual - 1e-12
        )

        valid = (
            valid
            &
            improves
        )

        if not np.any(
            valid
        ):
            break

        route_sign = np.where(
            choose_positive[
                rows,
                best_rail
            ],
            1,
            -1
        ).astype(
            np.int8
        )

        routes[
            rows[valid],
            term
        ] = (
            best_rail[
                valid
            ]
            + 1
        )

        signs[
            rows[valid],
            term
        ] = route_sign[
            valid
        ]

        selected_values = (
            rail_values[
                best_rail
            ]
        )

        residual[
            valid
        ] -= (
            route_sign[
                valid
            ]
            *
            selected_values[
                valid
            ]
        )

        used[
            rows[valid],
            best_rail[
                valid
            ]
        ] = True

        active_terms[
            valid
        ] += 1

    return (
        routes,
        signs,
        residual,
        active_terms
    )


# ============================================================
# ROUTE RECONSTRUCTION
# ============================================================


def compile_exact_routes_exhaustive(unique_bits: np.ndarray, rails: np.ndarray, max_terms: int):
    target_bits = [int(x) for x in unique_bits]
    pair_table = _build_pair_sum_table(rails)
    routes = {}
    if max_terms >= 1:
        for rid, bits in enumerate(rails):
            v = _bf16_to_float64_bits(int(bits))
            routes[int(float32_to_bf16_bits(v))] = ((rid, 1),)
            routes[int(float32_to_bf16_bits(-v))] = ((rid, -1),)
    if max_terms >= 2:
        for bits in target_bits:
            if bits in routes:
                continue
            tgt = _bf16_to_float64_bits(bits)
            cand = pair_table.get(tgt)
            if cand is None:
                continue
            if not _route_has_unique_rails(cand):
                continue
            if _exact_bf16_equal(bits, _route_value(cand, rails)):
                routes[bits] = cand
    if max_terms >= 3:
        rv = bf16_array_to_float32(rails).astype(np.float64)
        for bits in target_bits:
            if bits in routes:
                continue
            tgt = _bf16_to_float64_bits(bits)
            found = None
            for rid in range(len(rails)):
                rvv = float(rv[rid])
                rem = tgt - rvv
                pr = pair_table.get(rem)
                if pr is not None:
                    cand = pr + ((rid, 1),)
                    if _route_has_unique_rails(cand) and _exact_bf16_equal(bits, _route_value(cand, rails)):
                        found = cand
                        break
                rem = tgt + rvv
                pr = pair_table.get(rem)
                if pr is not None:
                    cand = pr + ((rid, -1),)
                    if _route_has_unique_rails(cand) and _exact_bf16_equal(bits, _route_value(cand, rails)):
                        found = cand
                        break
            if found is not None:
                routes[bits] = found
    if max_terms >= 4:
        pair_items = list(pair_table.items())
        for bits in target_bits:
            if bits in routes:
                continue
            tgt = _bf16_to_float64_bits(bits)
            found = None
            for va, ra in pair_items:
                comp = tgt - va
                rb = pair_table.get(comp)
                if rb is None:
                    continue
                cand = ra + rb
                if len(cand) == 0 or len(cand) > max_terms:
                    continue
                if not _route_has_unique_rails(cand):
                    continue
                if _exact_bf16_equal(bits, _route_value(cand, rails)):
                    found = cand
                    break
            if found is not None:
                routes[bits] = found
    return routes



def exhaustive_exact_count(unique_bits: np.ndarray, rails: np.ndarray, max_terms: int) -> int:
    table = compile_exact_routes_exhaustive(unique_bits, rails, max_terms)
    cnt = 0
    for b in unique_bits:
        if int(b) in table and table[int(b)] is not None:
            # double-check BF16 equality
            if _exact_bf16_equal(int(b), _route_value(table[int(b)], rails)):
                cnt += 1
    return cnt


# ============================================================
# MISSING VALUE REPAIR (exhaustive, monotone best-so-far)
# ============================================================


def _add_candidate(table: dict, value: float, route: tuple):
    if value not in table:
        table[value] = route



def _build_pair_sum_table(rails: np.ndarray):
    rail_values = bf16_array_to_float32(rails).astype(np.float64)
    table = {}
    table[0.0] = tuple()
    for i in range(len(rails)):
        vi = float(rail_values[i])
        _add_candidate(table, vi, ((i, 1),))
        _add_candidate(table, -vi, ((i, -1),))
    for i in range(len(rails)):
        vi = float(rail_values[i])
        for j in range(i + 1, len(rails)):
            vj = float(rail_values[j])
            _add_candidate(table, vi + vj, ((i, 1), (j, 1)))
            _add_candidate(table, vi - vj, ((i, 1), (j, -1)))
            _add_candidate(table, -vi + vj, ((i, -1), (j, 1)))
            _add_candidate(table, -vi - vj, ((i, -1), (j, -1)))
    return table



def _route_has_unique_rails(route: tuple) -> bool:
    ids = [int(t[0]) for t in route]
    return len(ids) == len(set(ids))



def _route_value(route: tuple, rails: np.ndarray) -> float:
    rv = bf16_array_to_float32(rails).astype(np.float64)
    total = 0.0
    for rid, sign in route:
        total += int(sign) * float(rv[rid])
    return total



def _bf16_to_float64_bits(bits: int) -> float:
    return float(bf16_bits_to_float32(int(bits)))



def _exact_bf16_equal(bits: int, value: float) -> bool:
    return int(bits) == int(float32_to_bf16_bits(value))



def reconstruct_routes(
    routes,
    signs,
    rails
):

    n = routes.shape[0]

    max_terms = routes.shape[1]

    rail_values = (
        bf16_array_to_float32(
            rails
        ).astype(
            np.float64
        )
    )

    reconstructed = np.zeros(
        n,
        dtype=np.float64
    )

    for term in range(
        max_terms
    ):

        ids = (
            routes[
                :,
                term
            ]
        )

        active = (
            ids > 0
        )

        if not np.any(
            active
        ):
            continue

        zero_based = (
            ids[
                active
            ]
            -
            1
        )

        reconstructed[
            active
        ] += (
            signs[
                active,
                term
            ].astype(
                np.float64
            )
            *
            rail_values[
                zero_based
            ]
        )

    return reconstructed


# ============================================================
# BF16 EXACTNESS
# ============================================================


def reconstructed_to_bf16(
    values
):

    values32 = np.asarray(
        values,
        dtype=np.float32
    )

    return fp32_array_to_bf16_bits(
        values32
    )



def exact_mask(
    target_bits,
    reconstructed_values
):

    reconstructed_bits = (
        reconstructed_to_bf16(
            reconstructed_values
        )
    )

    return (
        reconstructed_bits
        ==
        target_bits
    )


# ============================================================
# OBJECTIVE
# ============================================================


def calculate_objective(
    target_values,
    target_bits,
    counts,
    reconstructed_values
):

    exact = exact_mask(
        target_bits,
        reconstructed_values
    )

    diff = (
        target_values
        -
        reconstructed_values
    )

    weighted_mse = (
        np.sum(
            counts
            *
            (
                diff
                ** 2
            )
        )
        /
        np.sum(
            counts
        )
    )

    exact_unique = int(
        np.count_nonzero(
            exact
        )
    )

    weighted_exact = (
        float(
            np.sum(
                counts[
                    exact
                ]
            )
        )
        /
        float(
            np.sum(
                counts
            )
        )
    )

    return {
        "exact_mask": exact,

        "exact_unique": exact_unique,

        "weighted_exact": weighted_exact,

        "weighted_mse": float(
            weighted_mse
        ),

        "rmse": float(
            np.sqrt(
                weighted_mse
            )
        ),

        "max_error": float(
            np.max(
                np.abs(
                    diff
                )
            )
        ),
    }


# ============================================================
# EXHAUSTIVE EXACT ROUTE SEARCH (for fair evaluation)
# ============================================================
# Greedy routing is heuristic and underestimates true
# exact capacity (e.g., 64/4 greedy 395 vs exhaustive 2317).
# For final reporting we use the same exhaustive search
# as 03_bf16_rail_compile.py to fairly compare learned
# basis vs quantile baseline.
# ============================================================


