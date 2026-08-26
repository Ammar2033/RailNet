import json
import math
import mmap
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RAILNET-1B
# LEARNED BF16 SHARED BASIS COMPILER
#
# Target tensor:
#
#   model.layers.0.mlp.up_proj.weight
#
# Design:
#
#   BF16 values
#       ↓
#   shared BF16 rails
#       ↓
#   sparse {-1,0,+1} routes
#       ↓
#   weighted basis learning
#
# Runtime weight coefficients:
#   NONE
#
# Rail values:
#   BF16
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path(
    "model_data/model.safetensors"
)

TARGET_TENSOR = (
    "model.layers.0.mlp.up_proj.weight"
)

# Main experiments.
RAIL_COUNTS = [
    32,
    64,
    128,
]

MAX_TERMS_LIST = [
    2,
    3,
    4,
    6,
]

# Alternating optimization iterations.
MAX_ITERS = 8

# Weighted k-means initialization iterations.
INIT_KMEANS_ITERS = 8

# If weighted objective stops improving this much, stop.
CONVERGENCE_TOL = 1e-14

# Number of high-frequency residual candidates used
# when repairing the basis.
# Reduced from 256 to 64 for CPU time after PC crash reported.
RESIDUAL_CANDIDATES = 64

# Limit expensive candidate evaluation.
MAX_RAIL_REPAIRS_PER_ITER = 4

# Reserve first rail slots for min/max extreme values.
# NOTE: experiment showed overwriting rank-uniform slots with
# extremes DESTROYS dense-region pair coverage (4387 -> 4169).
# Disabled; tail gaps are handled by repair_missing_values().
EXTREME_RAIL_SLOTS = 0

# Post-learning exhaustive repair budget
# (max number of exhaustive compile calls during repair).
REPAIR_COMPILE_BUDGET = 96

# Safe-slot scan: only run when missing count is at most this.
SAFE_SCAN_MAX_MISSING = 48

# Safe-slot scan: max rails probed with sentinel compile.
SAFE_SCAN_LIMIT = 160

# Deterministic seed.
SEED = 42


# ============================================================
# FP32 / BF16 BIT HELPERS
# ============================================================

def bf16_bits_to_float32(
    bits: int
) -> np.float32:

    bits = int(bits)

    fp32_bits = (
        bits
        << 16
    )

    return np.float32(
        struct.unpack(
            "<f",
            struct.pack(
                "<I",
                fp32_bits
            )
        )[0]
    )


def bf16_array_to_float32(
    bits: np.ndarray
) -> np.ndarray:

    fp32_bits = (
        bits.astype(
            np.uint32
        )
        << 16
    )

    return fp32_bits.view(
        np.float32
    )


def float32_to_bf16_bits(
    value: float
) -> int:

    fp32_bits = struct.unpack(
        "<I",
        struct.pack(
            "<f",
            float(
                np.float32(value)
            )
        )
    )[0]

    return int(
        fp32_bits >> 16
    )


def fp32_array_to_bf16_bits(
    values: np.ndarray
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float32
    )

    bits = values.view(
        np.uint32
    )

    return (
        bits
        >>
        16
    ).astype(
        np.uint16
    )


def bf16_bitwise_equal(
    a: int,
    b: int
) -> bool:

    return int(a) == int(b)


# ============================================================
# SAFETENSORS
# ============================================================

def read_safetensors_header():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    with open(
        MODEL_FILE,
        "rb"
    ) as f:

        header_len_bytes = f.read(
            8
        )

        if len(
            header_len_bytes
        ) != 8:

            raise RuntimeError(
                "Invalid Safetensors file."
            )

        header_length = struct.unpack(
            "<Q",
            header_len_bytes
        )[0]

        header_bytes = f.read(
            header_length
        )

        if len(
            header_bytes
        ) != header_length:

            raise RuntimeError(
                "Incomplete Safetensors header."
            )

    header = json.loads(
        header_bytes.decode(
            "utf-8"
        )
    )

    return (
        header,
        8 + header_length
    )


def read_target_tensor():

    header, data_base_offset = (
        read_safetensors_header()
    )

    if TARGET_TENSOR not in header:

        raise KeyError(
            f"Tensor not found: "
            f"{TARGET_TENSOR}"
        )

    metadata = header[
        TARGET_TENSOR
    ]

    if metadata.get(
        "dtype"
    ) != "BF16":

        raise TypeError(
            "Target tensor is not BF16."
        )

    shape = tuple(
        int(x)
        for x in metadata[
            "shape"
        ]
    )

    offsets = metadata[
        "data_offsets"
    ]

    start = int(
        offsets[0]
    )

    end = int(
        offsets[1]
    )

    elements = math.prod(
        shape
    )

    byte_count = (
        end - start
    )

    expected_bytes = (
        elements * 2
    )

    if byte_count != expected_bytes:

        raise RuntimeError(
            "Tensor byte size mismatch."
        )

    absolute_start = (
        data_base_offset
        + start
    )

    with open(
        MODEL_FILE,
        "rb"
    ) as f:

        with mmap.mmap(
            f.fileno(),
            length=0,
            access=mmap.ACCESS_READ
        ) as mm:

            raw_bytes = mm[
                absolute_start:
                absolute_start + byte_count
            ]

    raw = np.frombuffer(
        raw_bytes,
        dtype=np.uint16
    ).copy()

    return (
        raw,
        shape
    )


# ============================================================
# UNIQUE VALUE DATA
# ============================================================

def analyze_unique_values(
    raw
):

    unique_values, counts = np.unique(
        raw,
        return_counts=True
    )

    values_float = (
        bf16_array_to_float32(
            unique_values
        ).astype(
            np.float64
        )
    )

    return (
        unique_values,
        counts.astype(
            np.float64
        ),
        values_float
    )


# ============================================================
# WEIGHTED 1D K-MEANS INITIALIZATION
# ============================================================

def initialize_rails(
    values_float,
    target_bits,
    counts,
    rail_count
):
    """
    Weighted 1D k-means-like initialization.

    Centers are always converted back to actual BF16 values.

    This is a better starting point than simple quantiles
    because frequent values have greater influence.

    Args:
        values_float: decoded BF16 values as float64 (length = unique)
        target_bits: original BF16 bits as uint16 (same length)
        counts: frequency of each unique value
        rail_count: desired number of rails
    """

    if rail_count >= len(values_float):

        selected = target_bits.copy()

        if len(selected) < rail_count:

            padding = np.zeros(
                rail_count - len(
                    selected
                ),
                dtype=np.uint16
            )

            selected = np.concatenate(
                [
                    selected,
                    padding
                ]
            )

        return selected[
            :rail_count
        ]

    # --------------------------------------------------------
    # Uniform quantile initialization.
    #
    # Previous weighted quantile + weighted k-means
    # concentrated rails near zero (dense region) and
    # missed tail values, giving poor exact coverage
    # (e.g., 64 rails: exhaustive 2460 vs uniform 4387).
    #
    # Uniform across sorted unique values gives better
    # spread and preserves tail representability while
    # still being frequency-aware via later weighted
    # optimization. This is a hardened fix.
    # --------------------------------------------------------

    order = np.argsort(
        values_float
    )

    sorted_values = (
        values_float[
            order
        ]
    )

    centers = []

    for i in range(
        rail_count
    ):

        # Uniform index across sorted unique values
        idx = int(
            (i + 0.5)
            / rail_count
            * len(sorted_values)
        )

        idx = min(
            idx,
            len(sorted_values) - 1
        )

        centers.append(
            float(
                sorted_values[
                    idx
                ]
            )
        )

    centers = np.asarray(
        centers,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Optional light weighted refinement (1 iteration)
    # is skipped to preserve spread. Full weighted
    # k-means would pull centers back to dense region
    # and degrade tail coverage. The coordinate-descent
    # `update_basis` later performs weighted optimization
    # anyway.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Convert every center to BF16.
    # --------------------------------------------------------

    rails = np.array(
        [
            float32_to_bf16_bits(
                center
            )
            for center in centers
        ],
        dtype=np.uint16
    )

    # --------------------------------------------------------
    # Remove duplicate BF16 rails.
    # --------------------------------------------------------

    rails = np.unique(
        rails
    )

    # --------------------------------------------------------
    # Need exactly rail_count slots.
    # Fill remaining slots with frequent actual values.
    # --------------------------------------------------------

    if len(rails) < rail_count:

        frequency_order = np.argsort(
            counts
        )[::-1]

        used = set(
            int(x)
            for x in rails
        )

        additions = []

        for index in frequency_order:

            candidate = int(
                target_bits[
                    index
                ]
            )

            if candidate in used:
                continue

            used.add(
                candidate
            )

            additions.append(
                candidate
            )

            if (
                len(rails)
                +
                len(additions)
                >= rail_count
            ):

                break

        if additions:

            rails = np.concatenate(
                [
                    rails,
                    np.array(
                        additions,
                        dtype=np.uint16
                    )
                ]
            )

    rails = rails[
        :rail_count
    ]

    # --------------------------------------------------------
    # Extreme tail rails.
    #
    # Rank-uniform spacing skips the extreme ends of the
    # sorted unique values (smallest denormal-like values
    # and largest magnitudes). Analysis showed these are
    # exactly the values missing from exact coverage.
    # Reserve first slots for min and max actual values.
    # --------------------------------------------------------

    if EXTREME_RAIL_SLOTS > 0 and rail_count >= 4:

        order_by_value = np.argsort(
            values_float
        )

        extreme_bits = []

        seen = set(
            int(x)
            for x in rails
        )

        for k in range(EXTREME_RAIL_SLOTS):

            low_index = int(
                order_by_value[k]
            )

            bits_low = int(
                target_bits[low_index]
            )

            if bits_low not in seen:
                extreme_bits.append(bits_low)
                seen.add(bits_low)

            high_index = int(
                order_by_value[-1 - k]
            )

            bits_high = int(
                target_bits[high_index]
            )

            if bits_high not in seen:
                extreme_bits.append(bits_high)
                seen.add(bits_high)

        for slot, bits_extreme in enumerate(
            extreme_bits
        ):

            rails[slot] = np.uint16(
                bits_extreme
            )

    return rails


# ============================================================
# FAST GREEDY ROUTING
# ============================================================

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

def repair_missing_values(
    target_values,
    target_bits,
    counts,
    rails,
    max_terms,
    verbose=False
):
    """
    Targeted post-learning repair.

    Analysis showed remaining missing values are extreme tail
    values skipped by rank-uniform init. Each missing value can
    often become 1-term exact by placing its own bits on an
    unused / least-used rail.

    Strategy:
        Phase 1 (batch): place top missing values into ALL
        strictly-unused rails at once, single compile check.
        Phase 2 (single): bounded swap trials on least-used rails.

    Accept only if exhaustive exact count improves.
    Monotone best-so-far. Hard-capped compile count for CPU time.
    """

    rails = rails.copy()

    total = len(target_bits)

    def compile_count(r):
        table = (
            compile_exact_routes_exhaustive(
                target_bits,
                r,
                max_terms
            )
        )
        c = 0
        for b in target_bits:
            if int(b) in table:
                c += 1
        return c, table

    compiles = 0

    best_count, _ = compile_count(rails)
    compiles += 1

    if best_count == total:
        return rails

    # --------------------------------------------------------
    # Rail usage histogram from current routes.
    # --------------------------------------------------------

    def usage_histogram(table):
        usage = np.zeros(
            len(rails),
            dtype=np.int64
        )
        for route in table.values():
            for rid, _sign in route:
                usage[rid] += 1
        return usage

    # --------------------------------------------------------
    # PHASE 1: batch placement into unused rails.
    # --------------------------------------------------------

    if compiles < REPAIR_COMPILE_BUDGET:

        _, table_now = (
            compile_count(rails)
        )
        compiles += 1

        usage = usage_histogram(
            table_now
        )

        zero_slots = [
            i for i in range(len(rails))
            if usage[i] == 0
        ]

        existing = set(
            int(x)
            for x in rails
        )

        missing_items = []

        for i in range(total):

            b = int(target_bits[i])

            if b not in table_now:
                missing_items.append(
                    (
                        int(counts[i]),
                        b
                    )
                )

        missing_items.sort(reverse=True)

        if zero_slots and missing_items:

            trial = rails.copy()

            placed = []

            for _cnt, cand_bits in missing_items:

                if not zero_slots:
                    break

                if cand_bits in existing:
                    continue

                slot = zero_slots.pop(0)

                trial[slot] = np.uint16(
                    cand_bits
                )

                existing.add(cand_bits)

                placed.append(slot)

            if placed:

                trial_count, trial_table = (
                    compile_count(trial)
                )

                compiles += 1

                if trial_count > best_count:

                    best_count = trial_count

                    rails = trial

                    if verbose:
                        print(
                            f"  [repair-batch] placed "
                            f"{len(placed)} missing -> exact "
                            f"{best_count}/{total}",
                            flush=True
                        )

        if best_count == total:
            return rails

    # --------------------------------------------------------
    # PHASE 2: single swap trials (bounded).
    # --------------------------------------------------------

    while compiles < REPAIR_COMPILE_BUDGET:

        _, table_now = compile_count(rails)
        compiles += 1

        current_count = sum(
            1
            for b in target_bits
            if int(b) in table_now
        )

        if current_count >= total:
            break

        usage = usage_histogram(
            table_now
        )

        slot_order = np.argsort(usage)[
            :8
        ]

        existing = set(
            int(x)
            for x in rails
        )

        missing_items = []

        for i in range(total):

            b = int(target_bits[i])

            if b not in table_now:
                missing_items.append(
                    (
                        int(counts[i]),
                        b
                    )
                )

        missing_items.sort(reverse=True)

        missing_items = (
            missing_items[:12]
        )

        improved = False

        for cand_slot in slot_order:

            if compiles >= REPAIR_COMPILE_BUDGET:
                break

            for _cnt, cand_bits in missing_items:

                if cand_bits in existing:
                    continue

                saved = int(
                    rails[cand_slot]
                )

                rails[cand_slot] = (
                    np.uint16(cand_bits)
                )

                trial_count, _ttable = (
                    compile_count(rails)
                )

                compiles += 1

                if trial_count > current_count:

                    current_count = trial_count

                    if current_count > best_count:
                        best_count = trial_count

                    existing.add(cand_bits)
                    existing.discard(saved)

                    improved = True

                    if verbose:
                        print(
                            f"  [repair] slot {cand_slot}: "
                            f"{saved:04X} -> {cand_bits:04X} "
                            f"exact {current_count}/{total}",
                            flush=True
                        )

                    break

                rails[cand_slot] = np.uint16(saved)

            if improved and compiles < REPAIR_COMPILE_BUDGET:
                continue

        if not improved:
            break

    return rails


# ============================================================
# SAFE SLOT REPAIR (final gap closer)
# ============================================================

def repair_safe_slots(
    target_values,
    target_bits,
    counts,
    rails,
    max_terms,
    verbose=False
):
    """
    Final-gap closer for the last few missing values.

    A rail slot is SAFE if removing it does not reduce the
    exact coverage (its own bit is still representable via
    other rails). Safe slots are free real estate: placing a
    missing value there is a guaranteed pure gain.

    Cost: one exhaustive compile per scanned rail. Bounded by
    SAFE_SCAN_LIMIT and only run when the missing count is
    small.
    """

    total = len(target_bits)

    original = rails.copy()

    def compile_count(r):
        table = (
            compile_exact_routes_exhaustive(
                target_bits,
                r,
                max_terms
            )
        )
        c = 0
        for b in target_bits:
            if int(b) in table:
                c += 1
        return c, table

    base_count, base_table = (
        compile_count(rails)
    )

    if base_count == total:
        return rails

    missing_items = []

    for i in range(total):

        b = int(target_bits[i])

        if b not in base_table:

            missing_items.append(
                (
                    int(counts[i]),
                    b
                )
            )

    if not missing_items:
        return rails

    # Only worth scanning when close to full coverage.
    if len(missing_items) > SAFE_SCAN_MAX_MISSING:
        return rails

    missing_items.sort(reverse=True)

    existing = set(
        int(x)
        for x in rails
    )

    # Sentinel far outside tensor range; combos with it can
    # never equal any target, so its slot is effectively empty.
    sentinel = np.uint16(0x4280)

    usage = np.zeros(
        len(rails),
        dtype=np.int64
    )

    for route in base_table.values():

        for rid, _sign in route:

            usage[rid] += 1

    scan_order = np.argsort(usage)

    safe_slots = []

    scanned = 0

    for slot in scan_order:

        if (
            len(safe_slots)
            >= len(missing_items)
        ):
            break

        if scanned >= SAFE_SCAN_LIMIT:
            break

        saved_bits = int(rails[slot])

        if saved_bits == int(sentinel):
            continue

        rails[slot] = sentinel

        trial_count, trial_table = (
            compile_count(rails)
        )

        scanned += 1

        if trial_count >= base_count:

            safe_slots.append(slot)

        rails[slot] = np.uint16(saved_bits)

    if not safe_slots:
        return rails

    # --------------------------------------------------------
    # Batch-place missing values into safe slots.
    # --------------------------------------------------------

    placed = 0

    for _cnt, cand_bits in missing_items:

        if not safe_slots:
            break

        if cand_bits in existing:
            continue

        slot = safe_slots.pop(0)

        rails[slot] = np.uint16(cand_bits)

        existing.add(cand_bits)

        placed += 1

    if placed == 0:
        return rails

    final_count, _ftable = compile_count(rails)

    if final_count < base_count:

        # Should not happen (safe slots are pure gains),
        # but keep monotone guarantee anyway.
        return original

    if verbose and placed:

        print(
            f"  [safe-repair] placed {placed} "
            f"missing into safe slots -> "
            f"{final_count}/{total}",
            flush=True
        )

    return rails


# ============================================================
# BASIS COORDINATE UPDATE
# ============================================================

def update_basis(
    target_values,
    counts,
    rails,
    routes,
    signs,
    exact_mask_values
):
    """
    Coordinate-descent update.

    For each rail, estimate the best shared rail value
    from all weights currently routed through that rail.

    The resulting value is quantized back to BF16.

    Important:
        - No per-weight coefficient is introduced.
        - Rails remain BF16 primitives.
        - `exact_mask_values` is kept in the signature for
          compatibility with the optimizer pipeline.
    """

    rail_count = len(
        rails
    )

    # Current BF16 rail values as float64 for optimization.
    rail_values = (
        bf16_array_to_float32(
            rails
        )
        .astype(
            np.float64
        )
    )

    max_terms = routes.shape[1]

    for rail_id in range(
        rail_count
    ):

        numerator = 0.0
        denominator = 0.0

        # ----------------------------------------------------
        # Find all route positions that use this rail.
        # ----------------------------------------------------

        for term in range(
            max_terms
        ):

            ids = routes[
                :,
                term
            ]

            active = (
                ids
                ==
                rail_id + 1
            )

            if not np.any(
                active
            ):
                continue

            active_indices = (
                np.flatnonzero(
                    active
                )
            )

            signs_active = (
                signs[
                    active_indices,
                    term
                ]
                .astype(
                    np.float64
                )
            )

            # ------------------------------------------------
            # Reconstruct the contribution of all OTHER rails
            # for these targets.
            # ------------------------------------------------

            current = np.zeros(
                len(active_indices),
                dtype=np.float64
            )

            for other_term in range(
                max_terms
            ):

                if other_term == term:
                    continue

                other_ids = routes[
                    active_indices,
                    other_term
                ]

                other_signs = signs[
                    active_indices,
                    other_term
                ].astype(
                    np.float64
                )

                other_active = (
                    other_ids
                    >
                    0
                )

                # Exclude the current rail explicitly.
                other_active &= (
                    other_ids
                    !=
                    rail_id + 1
                )

                if not np.any(
                    other_active
                ):
                    continue

                other_zero_based = (
                    other_ids[
                        other_active
                    ]
                    -
                    1
                )

                current[
                    other_active
                ] += (
                    other_signs[
                        other_active
                    ]
                    *
                    rail_values[
                        other_zero_based
                    ]
                )

            # ------------------------------------------------
            # Desired total contribution of this rail:
            #
            #   target = current + sign * rail
            #
            # therefore:
            #
            #   rail = sign * (target - current)
            # ------------------------------------------------

            desired = (
                target_values[
                    active_indices
                ]
                -
                current
            )

            signed_desired = (
                signs_active
                *
                desired
            )

            weights = (
                counts[
                    active_indices
                ]
            )

            numerator += float(
                np.sum(
                    weights
                    *
                    signed_desired
                )
            )

            denominator += float(
                np.sum(
                    weights
                )
            )

        # ----------------------------------------------------
        # No weight currently uses this rail.
        # ----------------------------------------------------

        if denominator <= 0.0:
            continue

        # ----------------------------------------------------
        # Weighted optimal value for this rail.
        # ----------------------------------------------------

        new_value = (
            numerator
            /
            denominator
        )

        # ----------------------------------------------------
        # Hardware primitive remains BF16.
        # ----------------------------------------------------

        new_bits = (
            float32_to_bf16_bits(
                new_value
            )
        )

        # ----------------------------------------------------
        # Update both representations.
        # ----------------------------------------------------

        rail_values[
            rail_id
        ] = (
            bf16_bits_to_float32(
                new_bits
            )
            .astype(
                np.float64
            )
        )

        rails[
            rail_id
        ] = np.uint16(
            new_bits
        )

    return rails

# ============================================================
# DUPLICATE RAIL REPAIR
# ============================================================

def repair_duplicate_rails(
    rails,
    target_values,
    counts,
    routes,
    signs,
    residual
):
    """
    If two learned rails collapse to the same BF16 value,
    replace the least useful duplicate rail with a high-frequency
    residual candidate.
    """

    seen = {}

    duplicates = []

    for i, value in enumerate(
        rails
    ):

        key = int(
            value
        )

        if key in seen:
            duplicates.append(
                i
            )
        else:
            seen[
                key
            ] = i

    if not duplicates:
        return rails

    # Residual candidates:
    # values not currently well represented.
    score = (
        counts
        *
        np.abs(
            residual
        )
    )

    order = np.argsort(
        score
    )[::-1]

    used = set(
        int(x)
        for x in rails
    )

    replacement_index = 0

    for rail_index in duplicates:

        while (
            replacement_index
            <
            len(order)
        ):

            candidate_value = (
                float(
                    target_values[
                        order[
                            replacement_index
                        ]
                    ]
                )
            )

            replacement_index += 1

            candidate_bits = (
                float32_to_bf16_bits(
                    candidate_value
                )
            )

            if candidate_bits in used:
                continue

            rails[
                rail_index
            ] = np.uint16(
                candidate_bits
            )

            used.add(
                candidate_bits
            )

            break

    return rails


# ============================================================
# LOCAL RAIL REPAIR
# ============================================================

def try_residual_repairs(
    target_values,
    target_bits,
    counts,
    rails,
    max_terms,
    current_best_score
):
    """
    Try a small number of high-value residual candidates.

    This is NOT brute force.

    Only a few rails are replaced and accepted if the objective
    improves.
    """

    (
        current_routes,
        current_signs,
        residual,
        _
    ) = greedy_routes(
        target_values,
        rails,
        max_terms
    )

    current_reconstructed = (
        reconstruct_routes(
            current_routes,
            current_signs,
            rails
        )
    )

    current_objective = (
        calculate_objective(
            target_values,
            target_bits,
            counts,
            current_reconstructed
        )
    )

    candidate_score = (
        current_objective[
            "exact_unique"
        ],
        current_objective[
            "weighted_exact"
        ],
        -current_objective[
            "weighted_mse"
        ]
    )

    if candidate_score > current_best_score:

        current_best_score = (
            candidate_score
        )

    residual_score = (
        counts
        *
        np.abs(
            residual
        )
    )

    order = np.argsort(
        residual_score
    )[::-1]

    candidate_indices = order[
        :min(
            RESIDUAL_CANDIDATES,
            len(order)
        )
    ]

    existing = set(
        int(x)
        for x in rails
    )

    # Least useful rails are those with low route frequency.
    # Filter zero entries (unused term slots) to avoid bincount
    # ValueError on negative values after sparse routing.
    active_mask = (
        current_routes > 0
    )

    if np.any(
        active_mask
    ):
        rail_usage = np.bincount(
            (
                current_routes[
                    active_mask
                ]
                - 1
            ),
            minlength=len(rails)
        )
    else:
        rail_usage = np.zeros(
            len(rails),
            dtype=np.int64
        )

    least_useful = np.argsort(
        rail_usage
    )[

        # Cap scanned rails: full scans on failing bases are
        # extremely slow (each candidate costs a greedy pass).
        :10
    ]

    accepted = 0

    best_rails = rails.copy()
    best_score = candidate_score

    for rail_index in least_useful:

        if (
            accepted
            >=
            MAX_RAIL_REPAIRS_PER_ITER
        ):
            break

        for candidate_index in candidate_indices:

            candidate_bits = int(
                target_bits[
                    candidate_index
                ]
            )

            if candidate_bits in existing:
                continue

            trial = rails.copy()

            trial[
                rail_index
            ] = np.uint16(
                candidate_bits
            )

            (
                trial_routes,
                trial_signs,
                _,
                _
            ) = greedy_routes(
                target_values,
                trial,
                max_terms
            )

            trial_reconstructed = (
                reconstruct_routes(
                    trial_routes,
                    trial_signs,
                    trial
                )
            )

            trial_objective = (
                calculate_objective(
                    target_values,
                    target_bits,
                    counts,
                    trial_reconstructed
                )
            )

            trial_score = (
                trial_objective[
                    "exact_unique"
                ],
                trial_objective[
                    "weighted_exact"
                ],
                -trial_objective[
                    "weighted_mse"
                ]
            )

            if trial_score > best_score:

                best_score = trial_score
                best_rails = trial
                accepted += 1
                # Keep existing in sync with best_rails
                existing = set(
                    int(x)
                    for x in best_rails
                )
                break

    return (
        best_rails,
        best_score
    )


# ============================================================
# SCORE
# ============================================================

def score_objective(
    objective
):
    """
    Spec section 29 ranking:

        1. exact_unique (full coverage first)
        2. weighted_exact
        3. -weighted_mse
    """

    return (
        int(
            objective[
                "exact_unique"
            ]
        ),
        float(
            objective[
                "weighted_exact"
            ]
        ),
        -float(
            objective[
                "weighted_mse"
            ]
        )
    )


# ============================================================
# LEARN ONE BASIS
# ============================================================

def learn_basis(
    target_values,
    target_bits,
    counts,
    rail_count,
    max_terms
):

    rails = initialize_rails(
        target_values,
        target_bits,
        counts,
        rail_count
    )

    best_rails = rails.copy()
    best_routes = None
    best_signs = None
    best_objective = None
    best_score = (
        -1.0,
        -1,
        -float("inf")
    )

    previous_mse = float(
        "inf"
    )

    history = []

    for iteration in range(
        1,
        MAX_ITERS + 1
    ):

        print(
            f"    [learn r={rail_count} t={max_terms}] "
            f"iter {iteration}/{MAX_ITERS}",
            flush=True
        )

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        (
            routes,
            signs,
            residual,
            active_terms
        ) = greedy_routes(
            target_values,
            rails,
            max_terms
        )

        # ----------------------------------------------------
        # Reconstruct
        # ----------------------------------------------------

        reconstructed = (
            reconstruct_routes(
                routes,
                signs,
                rails
            )
        )

        objective = (
            calculate_objective(
                target_values,
                target_bits,
                counts,
                reconstructed
            )
        )

        score = score_objective(
            objective
        )

        history.append(
            {
                "iteration": int(
                    iteration
                ),
                "exact_unique": int(
                    objective[
                        "exact_unique"
                    ]
                ),
                "weighted_exact": float(
                    objective[
                        "weighted_exact"
                    ]
                ),
                "weighted_mse": float(
                    objective[
                        "weighted_mse"
                    ]
                ),
                "max_error": float(
                    objective[
                        "max_error"
                    ]
                ),
                "active_terms": int(
                    np.sum(
                        active_terms
                    )
                ),
            }
        )

        # ----------------------------------------------------
        # Best-so-far
        # ----------------------------------------------------

        if score > best_score:

            best_score = score

            best_rails = rails.copy()

            best_routes = routes.copy()
            best_signs = signs.copy()

            best_objective = objective

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        current_mse = objective[
            "weighted_mse"
        ]

        improvement = (
            previous_mse
            -
            current_mse
        )

        previous_mse = current_mse

        if (
            iteration > 1
            and
            abs(improvement)
            <
            CONVERGENCE_TOL
        ):

            # Still allow basis repair once.
            pass

        # ----------------------------------------------------
        # Update basis
        # ----------------------------------------------------

        updated_rails = (
            update_basis(
                target_values,
                counts,
                rails.copy(),
                routes,
                signs,
                objective[
                    "exact_mask"
                ]
            )
        )

        updated_rails = (
            repair_duplicate_rails(
                updated_rails,
                target_values,
                counts,
                routes,
                signs,
                residual
            )
        )

        # ----------------------------------------------------
        # Evaluate new rails.
        # ----------------------------------------------------

        (
            next_routes,
            next_signs,
            next_residual,
            next_active
        ) = greedy_routes(
            target_values,
            updated_rails,
            max_terms
        )

        next_reconstructed = (
            reconstruct_routes(
                next_routes,
                next_signs,
                updated_rails
            )
        )

        next_objective = (
            calculate_objective(
                target_values,
                target_bits,
                counts,
                next_reconstructed
            )
        )

        next_score = score_objective(
            next_objective
        )

        # ----------------------------------------------------
        # Accept update only if it improves.
        # This gives monotonic best-so-far behavior.
        # ----------------------------------------------------

        if next_score >= score:

            rails = updated_rails

        else:

            # Even if weighted score doesn't improve,
            # replace with a basis that improves exact unique
            # coverage or frequency-weighted exact coverage.
            #
            # If neither improves, keep current basis.
            rails = rails

        # ----------------------------------------------------
        # Small residual repair every second iteration.
        # ----------------------------------------------------

        if (
            iteration % 2 == 0
        ):

            repaired_rails, repaired_score = (
                try_residual_repairs(
                    target_values,
                    target_bits,
                    counts,
                    rails,
                    max_terms,
                    best_score
                )
            )

            if repaired_score > score:

                rails = repaired_rails

        # ----------------------------------------------------
        # Full exact stop.
        # ----------------------------------------------------

        if (
            objective[
                "exact_unique"
            ]
            ==
            len(target_values)
        ):

            break

    # --------------------------------------------------------
    # Final targeted repair of missing values using
    # exhaustive evaluation (monotone best-so-far).
    # --------------------------------------------------------

    if best_rails is not None:

        repaired_rails = repair_missing_values(
            target_values,
            target_bits,
            counts,
            best_rails,
            max_terms
        )

        repaired_rails = repair_safe_slots(
            target_values,
            target_bits,
            counts,
            repaired_rails,
            max_terms
        )

        repaired_table = (
            compile_exact_routes_exhaustive(
                target_bits,
                repaired_rails,
                max_terms
            )
        )

        repaired_exact = sum(
            1
            for b in target_bits
            if int(b) in repaired_table
        )

        current_best_exact = (
            best_objective[
                "exact_unique"
            ]
            if best_objective is not None
            else -1
        )

        if (
            repaired_exact
            >=
            current_best_exact
        ):

            best_rails = repaired_rails

            # Refresh greedy route/objective on final rails so
            # returned representation stays consistent.
            (
                final_routes,
                final_signs,
                _res,
                _act
            ) = greedy_routes(
                target_values,
                best_rails,
                max_terms
            )

            final_reconstructed = (
                reconstruct_routes(
                    final_routes,
                    final_signs,
                    best_rails
                )
            )

            best_objective = (
                calculate_objective(
                    target_values,
                    target_bits,
                    counts,
                    final_reconstructed
                )
            )

            best_routes = final_routes
            best_signs = final_signs

    return {
        "rails": best_rails,
        "routes": best_routes,
        "signs": best_signs,
        "objective": best_objective,
        "score": best_score,
        "history": history,
    }


# ============================================================
# REPRESENTATION
# ============================================================

def representation_bits(
    rail_count,
    parameter_count,
    unique_count,
    routes,
    signs
):

    rail_bits = (
        rail_count
        *
        16
    )

    rail_index_bits = max(
        1,
        math.ceil(
            math.log2(
                rail_count
            )
        )
    )

    active_terms = int(
        np.count_nonzero(
            routes
        )
    )

    unique_route_bits = (
        active_terms
        *
        (
            rail_index_bits
            +
            1
        )
    )

    # Route ID into unique-value dictionary.
    unique_route_id_bits = max(
        1,
        math.ceil(
            math.log2(
                unique_count
            )
        )
    )

    full_tensor_route_id_bits = (
        parameter_count
        *
        unique_route_id_bits
    )

    original_bits = (
        parameter_count
        *
        16
    )

    # Full representation:
    #
    # rails
    # + route description for each unique value
    # + route ID for every tensor element
    full_bits = (
        rail_bits
        +
        unique_route_bits
        +
        full_tensor_route_id_bits
    )

    return {
        "rail_bits": int(
            rail_bits
        ),

        "rail_index_bits": int(
            rail_index_bits
        ),

        "active_terms": int(
            active_terms
        ),

        "unique_route_bits": int(
            unique_route_bits
        ),

        "unique_route_id_bits": int(
            unique_route_id_bits
        ),

        "full_tensor_route_id_bits": int(
            full_tensor_route_id_bits
        ),

        "full_representation_bits": int(
            full_bits
        ),

        "original_bits": int(
            original_bits
        ),

        "full_compression": float(
            original_bits
            /
            full_bits
            if full_bits > 0
            else 0.0
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Optional CLI override for batched benchmark runs:
    #
    #   python 04_bf16_learned_basis.py --rails 64,128 --terms 4 --tag b2
    # --------------------------------------------------------

    import sys

    global RAIL_COUNTS, MAX_TERMS_LIST

    output_tag = ""

    tensor_override = None

    args = sys.argv[1:]

    i = 0

    while i < len(args):

        if args[i] == "--rails" and i + 1 < len(args):

            RAIL_COUNTS = [
                int(x)
                for x in args[i + 1].split(",")
                if x
            ]

            i += 2

        elif args[i] == "--terms" and i + 1 < len(args):

            MAX_TERMS_LIST = [
                int(x)
                for x in args[i + 1].split(",")
                if x
            ]

            i += 2

        elif args[i] == "--tensor" and i + 1 < len(args):

            tensor_override = args[i + 1]

            # Default tag from tensor short name.
            short = (
                args[i + 1]
                .split(".")[-2]
            )

            output_tag = f"_{short}"

            i += 2

        elif args[i] == "--tag" and i + 1 < len(args):

            output_tag = f"_{args[i + 1]}"

            i += 2

        else:

            i += 1

    if tensor_override is not None:

        globals()["TARGET_TENSOR"] = (
            tensor_override
        )

    print("=" * 80)
    print(
        "RAILNET-1B BF16 LEARNED BASIS COMPILER"
    )
    print("=" * 80)

    print()
    print(
        f"Tensor      : {TARGET_TENSOR}"
    )

    print(
        f"Rails       : {RAIL_COUNTS}"
    )

    print(
        f"Max terms   : {MAX_TERMS_LIST}"
    )

    print(
        f"Iterations  : {MAX_ITERS}"
    )

    # --------------------------------------------------------
    # Read tensor.
    # --------------------------------------------------------

    start = time.perf_counter()

    raw, shape = (
        read_target_tensor()
    )

    read_seconds = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # Unique values.
    # --------------------------------------------------------

    (
        unique_bits,
        counts,
        unique_values
    ) = analyze_unique_values(
        raw
    )

    unique_count = len(
        unique_bits
    )

    parameter_count = len(
        raw
    )

    print()
    print(
        "MODEL"
    )

    print(
        "-" * 80
    )

    print(
        f"Shape               : {shape}"
    )

    print(
        f"Parameters          : "
        f"{parameter_count:,}"
    )

    print(
        f"Unique BF16 values  : "
        f"{unique_count:,}"
    )

    print(
        f"Unique ratio        : "
        f"{unique_count / parameter_count:.8%}"
    )

    print(
        f"Read time           : "
        f"{read_seconds:.4f}s"
    )

    # --------------------------------------------------------
    # Experiments.
    # --------------------------------------------------------

    results = []
    best_models = {}

    artifact_saved = False

    total_configs = (
        len(
            RAIL_COUNTS
        )
        *
        len(
            MAX_TERMS_LIST
        )
    )

    config_index = 0

    for rail_count in RAIL_COUNTS:

        for max_terms in MAX_TERMS_LIST:

            config_index += 1

            print()
            print("=" * 80)

            print(
                f"[{config_index}/{total_configs}] "
                f"RAILS={rail_count}, "
                f"MAX_TERMS={max_terms}"
            )

            print("=" * 80)

            start_config = (
                time.perf_counter()
            )

            learned = learn_basis(
                unique_values,
                unique_bits,
                counts,
                rail_count,
                max_terms
            )

            elapsed = (
                time.perf_counter()
                -
                start_config
            )

            objective = (
                learned[
                    "objective"
                ]
            )

            routes = (
                learned[
                    "routes"
                ]
            )

            signs = (
                learned[
                    "signs"
                ]
            )

            rails = (
                learned[
                    "rails"
                ]
            )

            exact_unique = int(
                objective[
                    "exact_unique"
                ]
            )

            weighted_exact = float(
                objective[
                    "weighted_exact"
                ]
            )

            full_exact = (
                exact_unique
                ==
                unique_count
            )

            representation = (
                representation_bits(
                    rail_count,
                    parameter_count,
                    unique_count,
                    routes,
                    signs
                )
            )

            # ------------------------------------------------
            # Exhaustive evaluation for fair baseline compare
            # ------------------------------------------------
            exhaustive_exact = exhaustive_exact_count(
                unique_bits,
                rails,
                max_terms
            )

            exhaustive_ratio = (
                exhaustive_exact / unique_count
                if unique_count
                else 0.0
            )

            print()
            print(
                f"Exact unique (greedy) : "
                f"{exact_unique:,} / "
                f"{unique_count:,} "
                f"({exact_unique / unique_count:.2%})"
            )

            print(
                f"Exact unique (exhaust.) : "
                f"{exhaustive_exact:,} / "
                f"{unique_count:,} "
                f"({exhaustive_ratio:.2%})"
            )

            print(
                f"Weighted exact      : "
                f"{weighted_exact:.8%}"
            )

            print(
                f"Weighted RMSE       : "
                f"{objective['rmse']:.8e}"
            )

            print(
                f"Max error           : "
                f"{objective['max_error']:.8e}"
            )

            print(
                f"Active route terms  : "
                f"{representation['active_terms']:,}"
            )

            print(
                f"Full exact          : "
                f"{bool(full_exact)}"
            )

            print(
                f"Full representation: "
                f"{representation['full_representation_bits']:,} bits"
            )

            print(
                f"Original            : "
                f"{representation['original_bits']:,} bits"
            )

            print(
                f"Full compression    : "
                f"{representation['full_compression']:.4f}x"
            )

            print(
                f"Runtime             : "
                f"{elapsed:.4f}s"
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            result = {
                "rails": int(
                    rail_count
                ),

                "max_terms": int(
                    max_terms
                ),

                "unique_values": int(
                    unique_count
                ),

                "parameters": int(
                    parameter_count
                ),

                "exact_unique": int(
                    exact_unique
                ),

                "missing_unique": int(
                    unique_count
                    -
                    exact_unique
                ),

                "exact_ratio": float(
                    exact_unique
                    /
                    unique_count
                ),

                "weighted_exact": float(
                    weighted_exact
                ),

                "weighted_rmse": float(
                    objective[
                        "rmse"
                    ]
                ),

                "weighted_mse": float(
                    objective[
                        "weighted_mse"
                    ]
                ),

                "max_error": float(
                    objective[
                        "max_error"
                    ]
                ),

                "active_terms": int(
                    representation[
                        "active_terms"
                    ]
                ),

                "exhaustive_exact": int(
                    exhaustive_exact
                ),

                "exhaustive_ratio": float(
                    exhaustive_ratio
                ),

                # Spec 29: full_tensor_exact is judged by
                # exhaustive representability (a valid route
                # EXISTS). Greedy router failure is a search
                # limitation, not a representation limitation,
                # and is reported separately as exact_unique.
                "full_tensor_exact": bool(
                    exhaustive_exact == unique_count
                ),

                "greedy_full_tensor": bool(
                    full_exact
                ),

                "rail_bits": int(
                    representation[
                        "rail_bits"
                    ]
                ),

                "unique_route_bits": int(
                    representation[
                        "unique_route_bits"
                    ]
                ),

                "full_tensor_route_id_bits": int(
                    representation[
                        "full_tensor_route_id_bits"
                    ]
                ),

                "full_representation_bits": int(
                    representation[
                        "full_representation_bits"
                    ]
                ),

                "original_bits": int(
                    representation[
                        "original_bits"
                    ]
                ),

                "full_compression": float(
                    representation[
                        "full_compression"
                    ]
                ),

                "runtime_seconds": float(
                    elapsed
                ),

                "iterations": int(
                    len(
                        learned[
                            "history"
                        ]
                    )
                ),
            }

            results.append(
                result
            )

            best_models[
                (
                    rail_count,
                    max_terms
                )
            ] = learned

            # ------------------------------------------------
            # Save artifacts of the first lossless config for
            # the runtime exact-kernel oracle test (05).
            # ------------------------------------------------

            if (
                result[
                    "full_tensor_exact"
                ]
                and not artifact_saved
            ):

                artifact_table = (
                    compile_exact_routes_exhaustive(
                        unique_bits,
                        rails,
                        max_terms
                    )
                )

                routes_json = {
                    str(bits): [
                        [
                            int(rid),
                            int(sgn)
                        ]
                        for rid, sgn in route
                    ]
                    for bits, route in artifact_table.items()
                }

                artifact = {
                    "tensor": TARGET_TENSOR,

                    "rails": int(rail_count),

                    "max_terms": int(max_terms),

                    "rail_bits": [
                        int(b)
                        for b in rails
                    ],

                    "routes": routes_json,

                    "exact_unique": int(
                        exhaustive_exact
                    ),

                    "unique_values": int(
                        unique_count
                    ),
                }

                with open(
                    f"railnet_lossless_basis{output_tag}.json",
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        artifact,
                        f
                    )

                artifact_saved = True

                print(
                    f"LOSSLESS BASIS SAVED: "
                    f"railnet_lossless_basis{output_tag}.json",
                    flush=True
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 80)

    dataframe = pd.DataFrame(
        results
    )

    columns = [
        "rails",
        "max_terms",
        "exact_unique",
        "exact_ratio",
        "exhaustive_exact",
        "exhaustive_ratio",
        "weighted_exact",
        "weighted_rmse",
        "max_error",
        "active_terms",
        "full_tensor_exact",
        "greedy_full_tensor",
        "full_compression",
        "runtime_seconds",
    ]

    print(
        dataframe[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST COVERAGE
    # ========================================================

    best_coverage = (
        dataframe.sort_values(
            [
                "weighted_exact",
                "exact_unique",
                "full_compression"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .iloc[0]
    )

    print()
    print("=" * 80)
    print(
        "BEST WEIGHTED COVERAGE"
    )
    print("=" * 80)

    print(
        best_coverage.to_string()
    )

    # ========================================================
    # BEST EXACT COVERAGE
    # ========================================================

    best_exact_unique = (
        dataframe.sort_values(
            [
                "exact_unique",
                "weighted_exact"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )

    print()
    print("=" * 80)
    print(
        "BEST EXACT-UNIQUE COVERAGE"
    )
    print("=" * 80)

    print(
        best_exact_unique.to_string()
    )

    # ========================================================
    # LOSSLESS
    # ========================================================

    lossless = dataframe[
        dataframe[
            "full_tensor_exact"
        ]
    ]

    print()
    print("=" * 80)
    print(
        "LOSSLESS CONFIGURATIONS"
    )
    print("=" * 80)

    if len(lossless):

        print(
            lossless[
                columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "NONE"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    dataframe.to_csv(
        f"railnet_bf16_learned_basis{output_tag}.csv",
        index=False
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_results = []

    for result in results:

        json_results.append(
            {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (np.floating,)
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (np.integer,)
                    )
                    else bool(value)
                    if isinstance(
                        value,
                        (np.bool_,)
                    )
                    else value
                )
                for key, value in result.items()
            }
        )

    output = {
        "tensor": TARGET_TENSOR,

        "shape": [
            int(x)
            for x in shape
        ],

        "parameters": int(
            parameter_count
        ),

        "unique_values": int(
            unique_count
        ),

        "results": json_results,
    }

    with open(
        f"railnet_bf16_learned_basis{output_tag}.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 80)
    print(
        "FILES SAVED"
    )
    print("=" * 80)

    print(
        f"railnet_bf16_learned_basis{output_tag}.csv"
    )

    print(
        f"railnet_bf16_learned_basis{output_tag}.json"
    )

    print()
    print("=" * 80)
    print(
        "V3 LEARNED BASIS EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()