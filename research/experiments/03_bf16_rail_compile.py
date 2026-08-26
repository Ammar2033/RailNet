import json
import math
import mmap
import struct
import time
from pathlib import Path

import numpy as np


# ============================================================
# RailNet-1B
# BF16 SHARED RAIL COMPILER
#
# Target:
#   model.layers.0.mlp.up_proj.weight
#
# Shape:
#   (6912, 1152)
#
# Parameters:
#   7,962,624
#
# Strategy:
#
#   BF16 tensor
#       ↓
#   unique BF16 values
#       ↓
#   shared rail basis
#       ↓
#   {-1, 0, +1} topology
#       ↓
#   exact route search
#
# IMPORTANT:
#   No per-weight floating-point coefficient.
#   Rails themselves are BF16 primitives.
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

# Tested shared primitive counts.
RAIL_COUNTS = [
    16,
    32,
    64,
]

# Maximum number of distinct rails per weight.
MAX_TERMS_LIST = [
    1,
    2,
    3,
    4,
]

# Top frequent BF16 values considered during
# frequency-aware rail selection.
TOP_ANCHORS = 2048

# Used only for reporting.
FULL_TENSOR_PARAMETERS = 7_962_624


# ============================================================
# FILE / SIZE HELPERS
# ============================================================

def human_size(nbytes: int) -> str:
    value = float(nbytes)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


# ============================================================
# BF16 HELPERS
# ============================================================

def bf16_array_to_fp32(
    bf16_values: np.ndarray
) -> np.ndarray:

    fp32_bits = (
        bf16_values.astype(
            np.uint32
        )
        << 16
    )

    return fp32_bits.view(
        np.float32
    )


def bf16_bits_to_fp32(
    bits: int
) -> np.float32:

    fp32_bits = (
        int(bits)
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


def bf16_to_float64(
    bits: int
) -> float:

    return float(
        bf16_bits_to_fp32(
            bits
        )
    )


def fp32_to_bf16_bits(
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

    return (
        fp32_bits
        >>
        16
    )


def exact_bf16_equal(
    original_bits: int,
    reconstructed_value: float
) -> bool:

    reconstructed_bits = (
        fp32_to_bf16_bits(
            reconstructed_value
        )
    )

    return (
        int(original_bits)
        ==
        int(reconstructed_bits)
    )


# ============================================================
# SAFETENSORS HEADER
# ============================================================

def read_safetensors_header():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_FILE}"
        )

    with open(
        MODEL_FILE,
        "rb"
    ) as f:

        raw_length = f.read(8)

        if len(raw_length) != 8:
            raise RuntimeError(
                "Geçersiz Safetensors dosyası."
            )

        header_length = struct.unpack(
            "<Q",
            raw_length
        )[0]

        header_bytes = f.read(
            header_length
        )

        if len(header_bytes) != header_length:
            raise RuntimeError(
                "Safetensors header eksik."
            )

    header = json.loads(
        header_bytes.decode(
            "utf-8"
        )
    )

    data_base_offset = (
        8
        +
        header_length
    )

    return (
        header,
        data_base_offset
    )


# ============================================================
# READ BF16 TARGET TENSOR
# ============================================================

def read_bf16_tensor():

    header, data_base_offset = (
        read_safetensors_header()
    )

    if TARGET_TENSOR not in header:
        raise KeyError(
            f"Tensor bulunamadı: {TARGET_TENSOR}"
        )

    metadata = header[
        TARGET_TENSOR
    ]

    dtype = metadata.get(
        "dtype"
    )

    if dtype != "BF16":
        raise TypeError(
            f"Beklenen dtype BF16, gelen {dtype}"
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

    if (
        not isinstance(offsets, list)
        or
        len(offsets) != 2
    ):
        raise RuntimeError(
            "Geçersiz data_offsets."
        )

    start = int(
        offsets[0]
    )

    end = int(
        offsets[1]
    )

    element_count = math.prod(
        shape
    )

    byte_count = (
        end
        -
        start
    )

    expected_bytes = (
        element_count
        *
        2
    )

    if byte_count != expected_bytes:
        raise RuntimeError(
            "Tensor byte boyutu uyuşmuyor.\n"
            f"Beklenen: {expected_bytes}\n"
            f"Gerçek  : {byte_count}"
        )

    absolute_start = (
        data_base_offset
        +
        start
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
                absolute_start
                +
                byte_count
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
# UNIQUE VALUES
# ============================================================

def unique_values_with_counts(
    raw: np.ndarray
):

    values, counts = np.unique(
        raw,
        return_counts=True
    )

    # Most frequent first.
    order = np.argsort(
        counts
    )[::-1]

    values = values[
        order
    ]

    counts = counts[
        order
    ]

    return (
        values,
        counts
    )


# ============================================================
# RAIL SELECTION
# ============================================================

def select_rails(
    unique_values: np.ndarray,
    counts: np.ndarray,
    rail_count: int
) -> np.ndarray:
    """
    Frequency-aware quantile selection.

    This is only the first basis-selection strategy.
    It is intentionally deterministic.
    """

    top_n = min(
        TOP_ANCHORS,
        len(unique_values)
    )

    candidates = (
        unique_values[
            :top_n
        ]
    )

    candidate_counts = (
        counts[
            :top_n
        ]
    )

    candidate_fp64 = (
        bf16_array_to_fp32(
            candidates
        ).astype(
            np.float64
        )
    )

    order = np.argsort(
        candidate_fp64
    )

    sorted_values = (
        candidates[
            order
        ]
    )

    sorted_counts = (
        candidate_counts[
            order
        ]
    )

    cumulative = np.cumsum(
        sorted_counts
    )

    total = int(
        cumulative[-1]
    )

    selected = []

    for i in range(
        rail_count
    ):

        if rail_count == 1:
            target = 0
        else:
            target = (
                total
                *
                i
                /
                (
                    rail_count
                    -
                    1
                )
            )

        index = int(
            np.searchsorted(
                cumulative,
                target,
                side="left"
            )
        )

        index = min(
            index,
            len(sorted_values) - 1
        )

        value = int(
            sorted_values[index]
        )

        if value not in selected:
            selected.append(
                value
            )

    # Fill missing rail slots deterministically.
    if len(selected) < rail_count:

        all_values = np.unique(
            unique_values
        )

        for value in all_values:

            value_int = int(
                value
            )

            if value_int in selected:
                continue

            selected.append(
                value_int
            )

            if len(selected) >= rail_count:
                break

    return np.array(
        selected[
            :rail_count
        ],
        dtype=np.uint16
    )


# ============================================================
# PAIR-SUM TABLE
# ============================================================

def add_candidate(
    table: dict,
    value: float,
    route: tuple
):
    """
    Store first route for a numerical value.
    """

    if value not in table:
        table[
            value
        ] = route


def build_pair_sum_table(
    rails: np.ndarray
):
    """
    Create all one/two-term signed combinations.

    For N rails:
        O(N²)

    N is only 16/32/64, so this is tiny.
    """

    rail_values = (
        bf16_array_to_fp32(
            rails
        ).astype(
            np.float64
        )
    )

    table = {}

    # Zero.
    table[
        0.0
    ] = tuple()

    # One-term routes.
    for i in range(
        len(rails)
    ):

        value = (
            float(
                rail_values[i]
            )
        )

        add_candidate(
            table,
            value,
            (
                (
                    i,
                    1
                ),
            )
        )

        add_candidate(
            table,
            -value,
            (
                (
                    i,
                    -1
                ),
            )
        )

    # Two-term routes.
    for i in range(
        len(rails)
    ):

        vi = float(
            rail_values[i]
        )

        for j in range(
            i + 1,
            len(rails)
        ):

            vj = float(
                rail_values[j]
            )

            add_candidate(
                table,
                vi + vj,
                (
                    (
                        i,
                        1
                    ),
                    (
                        j,
                        1
                    ),
                )
            )

            add_candidate(
                table,
                vi - vj,
                (
                    (
                        i,
                        1
                    ),
                    (
                        j,
                        -1
                    ),
                )
            )

            add_candidate(
                table,
                -vi + vj,
                (
                    (
                        i,
                        -1
                    ),
                    (
                        j,
                        1
                    ),
                )
            )

            add_candidate(
                table,
                -vi - vj,
                (
                    (
                        i,
                        -1
                    ),
                    (
                        j,
                        -1
                    ),
                )
            )

    return table


# ============================================================
# ROUTE VALIDATION
# ============================================================

def route_has_unique_rails(
    route: tuple
) -> bool:

    rail_ids = [
        int(
            term[0]
        )
        for term in route
    ]

    return (
        len(rail_ids)
        ==
        len(set(rail_ids))
    )


def route_value(
    route: tuple,
    rails: np.ndarray
) -> float:

    rail_values = (
        bf16_array_to_fp32(
            rails
        ).astype(
            np.float64
        )
    )

    total = 0.0

    for rail_id, sign in route:

        total += (
            int(sign)
            *
            float(
                rail_values[
                    rail_id
                ]
            )
        )

    return total


# ============================================================
# EXACT ROUTE SEARCH
# ============================================================

def compile_exact_routes(
    unique_values: np.ndarray,
    rails: np.ndarray,
    max_terms: int
):
    """
    Search exact signed sparse routes.

    Supported:
        1 term
        2 terms
        3 terms
        4 terms

    All route components use distinct rails.

    Exactness is checked at BF16 level after reconstruction.
    """

    target_bits = [
        int(
            x
        )
        for x in unique_values
    ]

    target_values = [
        bf16_to_float64(
            bits
        )
        for bits in target_bits
    ]

    pair_table = (
        build_pair_sum_table(
            rails
        )
    )

    routes = {}

    # --------------------------------------------------------
    # 1 TERM
    # --------------------------------------------------------

    if max_terms >= 1:

        for rail_id, bits in enumerate(
            rails
        ):

            value = bf16_to_float64(
                int(bits)
            )

            positive_route = (
                (
                    rail_id,
                    1
                ),
            )

            negative_route = (
                (
                    rail_id,
                    -1
                ),
            )

            positive_bits = (
                fp32_to_bf16_bits(
                    value
                )
            )

            negative_bits = (
                fp32_to_bf16_bits(
                    -value
                )
            )

            routes[
                positive_bits
            ] = positive_route

            routes[
                negative_bits
            ] = negative_route

    # --------------------------------------------------------
    # 2 TERM
    # --------------------------------------------------------

    if max_terms >= 2:

        for bits in target_bits:

            if bits in routes:
                continue

            target = bf16_to_float64(
                bits
            )

            candidate = (
                pair_table.get(
                    target
                )
            )

            if candidate is None:
                continue

            if not route_has_unique_rails(
                candidate
            ):
                continue

            reconstructed = route_value(
                candidate,
                rails
            )

            if exact_bf16_equal(
                bits,
                reconstructed
            ):
                routes[
                    bits
                ] = candidate

    # --------------------------------------------------------
    # 3 TERM
    # --------------------------------------------------------

    if max_terms >= 3:

        rail_values = (
            bf16_array_to_fp32(
                rails
            ).astype(
                np.float64
            )
        )

        for bits in target_bits:

            if bits in routes:
                continue

            target = bf16_to_float64(
                bits
            )

            found = None

            for rail_id in range(
                len(rails)
            ):

                rv = float(
                    rail_values[
                        rail_id
                    ]
                )

                # target = pair + (+rail)
                remainder = (
                    target
                    -
                    rv
                )

                pair_route = (
                    pair_table.get(
                        remainder
                    )
                )

                if pair_route is not None:

                    candidate = (
                        pair_route
                        +
                        (
                            (
                                rail_id,
                                1
                            ),
                        )
                    )

                    if route_has_unique_rails(
                        candidate
                    ):

                        reconstructed = route_value(
                            candidate,
                            rails
                        )

                        if exact_bf16_equal(
                            bits,
                            reconstructed
                        ):
                            found = candidate
                            break

                # target = pair + (-rail)
                remainder = (
                    target
                    +
                    rv
                )

                pair_route = (
                    pair_table.get(
                        remainder
                    )
                )

                if pair_route is not None:

                    candidate = (
                        pair_route
                        +
                        (
                            (
                                rail_id,
                                -1
                            ),
                        )
                    )

                    if route_has_unique_rails(
                        candidate
                    ):

                        reconstructed = route_value(
                            candidate,
                            rails
                        )

                        if exact_bf16_equal(
                            bits,
                            reconstructed
                        ):
                            found = candidate
                            break

            if found is not None:
                routes[
                    bits
                ] = found

    # --------------------------------------------------------
    # 4 TERM
    #
    # target = pair_a + pair_b
    # --------------------------------------------------------

    if max_terms >= 4:

        pair_items = list(
            pair_table.items()
        )

        # Pair table includes zero and one-term routes.
        # That is useful because a 4-term search may
        # actually find a lower-term representation first.
        for bits in target_bits:

            if bits in routes:
                continue

            target = bf16_to_float64(
                bits
            )

            found = None

            for pair_a_value, route_a in pair_items:

                complement = (
                    target
                    -
                    pair_a_value
                )

                route_b = (
                    pair_table.get(
                        complement
                    )
                )

                if route_b is None:
                    continue

                candidate = (
                    route_a
                    +
                    route_b
                )

                # Remove empty route.
                if len(candidate) == 0:
                    continue

                if len(candidate) > max_terms:
                    continue

                if not route_has_unique_rails(
                    candidate
                ):
                    continue

                reconstructed = route_value(
                    candidate,
                    rails
                )

                if exact_bf16_equal(
                    bits,
                    reconstructed
                ):

                    found = candidate
                    break

            if found is not None:
                routes[
                    bits
                ] = found

    return routes


# ============================================================
# ROUTE TABLE
# ============================================================

def build_route_table_for_unique_values(
    unique_values: np.ndarray,
    rails: np.ndarray,
    max_terms: int
):
    route_map = compile_exact_routes(
        unique_values,
        rails,
        max_terms
    )

    compiled = {}

    exact_count = 0

    for bits in unique_values:

        bits_int = int(
            bits
        )

        route = route_map.get(
            bits_int
        )

        if route is None:

            compiled[
                bits_int
            ] = None

            continue

        reconstructed = route_value(
            route,
            rails
        )

        if exact_bf16_equal(
            bits_int,
            reconstructed
        ):

            compiled[
                bits_int
            ] = route

            exact_count += 1

        else:

            compiled[
                bits_int
            ] = None

    return (
        compiled,
        int(exact_count)
    )


# ============================================================
# ROUTE STATISTICS
# ============================================================

def route_statistics(
    route_table: dict,
    exact_count: int
):

    exact_routes = [
        route
        for route in route_table.values()
        if route is not None
    ]

    route_count = len(
        route_table
    )

    missing = (
        route_count
        -
        int(exact_count)
    )

    term_counts = [
        len(route)
        for route in exact_routes
    ]

    exact_ratio = (
        int(exact_count)
        /
        route_count
        if route_count > 0
        else 0.0
    )

    return {
        "exact_unique": int(
            exact_count
        ),

        "missing_unique": int(
            missing
        ),

        "exact_ratio": float(
            exact_ratio
        ),

        "average_terms": float(
            np.mean(
                term_counts
            )
            if term_counts
            else 0.0
        ),

        "max_terms": int(
            max(
                term_counts
            )
            if term_counts
            else 0
        ),
    }


# ============================================================
# FULL TENSOR ROUTE CHECK
# ============================================================

def full_tensor_exact_check(
    raw: np.ndarray,
    route_table: dict
) -> bool:
    """
    Full tensor is exact iff EVERY unique BF16 value
    has an exact route.
    """

    unique_raw = np.unique(
        raw
    )

    for bits in unique_raw:

        if route_table.get(
            int(bits)
        ) is None:
            return False

    return True


# ============================================================
# FULL TENSOR ROUTE EXPANSION
# ============================================================

def expand_routes(
    raw: np.ndarray,
    route_table: dict
):
    """
    Convert every BF16 tensor element into a route ID.

    This should only be called for an exact configuration.
    """

    valid_items = [
        (
            int(bits),
            route
        )
        for bits, route in route_table.items()
        if route is not None
    ]

    if not valid_items:

        return (
            np.full(
                len(raw),
                -1,
                dtype=np.int32
            ),
            []
        )

    valid_bits = np.array(
        [
            item[0]
            for item in valid_items
        ],
        dtype=np.uint16
    )

    route_list = [
        item[1]
        for item in valid_items
    ]

    route_ids = np.arange(
        len(route_list),
        dtype=np.int32
    )

    order = np.argsort(
        valid_bits
    )

    sorted_bits = (
        valid_bits[
            order
        ]
    )

    sorted_ids = (
        route_ids[
            order
        ]
    )

    positions = np.searchsorted(
        sorted_bits,
        raw
    )

    valid = (
        positions
        <
        len(sorted_bits)
    )

    safe_positions = np.minimum(
        positions,
        len(sorted_bits) - 1
    )

    valid &= (
        sorted_bits[
            safe_positions
        ]
        ==
        raw
    )

    output = np.full(
        len(raw),
        -1,
        dtype=np.int32
    )

    if np.any(valid):

        output[
            valid
        ] = (
            sorted_ids[
                positions[
                    valid
                ]
            ]
        )

    return (
        output,
        route_list
    )


# ============================================================
# REPRESENTATION ESTIMATE
# ============================================================

def representation_estimate(
    rail_count: int,
    route_table: dict,
    full_tensor_parameter_count: int
):
    """
    Two representations are reported.

    1. UNIQUE DICTIONARY:
       Rails + route for each unique BF16 value.

    2. FULL TENSOR:
       Rails + route ID for every tensor element.

    These are intentionally separated.
    """

    rail_bits = (
        int(rail_count)
        *
        16
    )

    route_index_bits = max(
        1,
        int(
            math.ceil(
                math.log2(
                    rail_count
                )
            )
        )
    )

    # --------------------------------------------------------
    # Unique dictionary route bits.
    # --------------------------------------------------------

    unique_route_bits = 0
    unique_active_terms = 0

    for route in route_table.values():

        if route is None:
            continue

        unique_active_terms += len(
            route
        )

        unique_route_bits += (
            len(route)
            *
            (
                route_index_bits
                +
                1
            )
        )

    unique_total_bits = (
        rail_bits
        +
        unique_route_bits
    )

    # --------------------------------------------------------
    # Full tensor route IDs.
    #
    # Each weight points to a route ID.
    # --------------------------------------------------------

    unique_route_count = sum(
        route is not None
        for route in route_table.values()
    )

    route_id_bits = max(
        1,
        int(
            math.ceil(
                math.log2(
                    max(
                        unique_route_count,
                        1
                    )
                )
            )
        )
    )

    full_route_bits = (
        int(full_tensor_parameter_count)
        *
        route_id_bits
    )

    full_total_bits = (
        rail_bits
        +
        unique_route_bits
        +
        full_route_bits
    )

    original_bits = (
        int(full_tensor_parameter_count)
        *
        16
    )

    return {
        "rail_bits": int(
            rail_bits
        ),

        "route_index_bits": int(
            route_index_bits
        ),

        "unique_route_bits": int(
            unique_route_bits
        ),

        "unique_active_terms": int(
            unique_active_terms
        ),

        "unique_total_bits": int(
            unique_total_bits
        ),

        "full_route_id_bits": int(
            route_id_bits
        ),

        "full_tensor_route_bits": int(
            full_route_bits
        ),

        "full_total_bits": int(
            full_total_bits
        ),

        "original_bits": int(
            original_bits
        ),

        "unique_compression": float(
            original_bits
            /
            unique_total_bits
            if unique_total_bits > 0
            else 0.0
        ),

        "full_compression": float(
            original_bits
            /
            full_total_bits
            if full_total_bits > 0
            else 0.0
        ),
    }


# ============================================================
# ROUTE FILE SAVE
# ============================================================

def make_json_safe(value):

    if isinstance(
        value,
        (np.integer,)
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,)
    ):
        return float(value)

    if isinstance(
        value,
        (np.bool_,)
    ):
        return bool(value)

    if isinstance(
        value,
        np.ndarray
    ):
        return value.tolist()

    if isinstance(
        value,
        dict
    ):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        list
    ):
        return [
            make_json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "RAILNET-1B BF16 SHARED RAIL COMPILER"
    )
    print("=" * 80)

    print()
    print(
        f"Tensor: {TARGET_TENSOR}"
    )

    # --------------------------------------------------------
    # Read tensor
    # --------------------------------------------------------

    start = time.perf_counter()

    raw, shape = (
        read_bf16_tensor()
    )

    read_seconds = (
        time.perf_counter()
        -
        start
    )

    parameter_count = len(
        raw
    )

    print()
    print(
        f"Tensor shape       : {shape}"
    )

    print(
        f"Parameters         : "
        f"{parameter_count:,}"
    )

    print(
        f"Read time          : "
        f"{read_seconds:.4f}s"
    )

    # --------------------------------------------------------
    # Unique analysis
    # --------------------------------------------------------

    unique_values, counts = (
        unique_values_with_counts(
            raw
        )
    )

    unique_count = len(
        unique_values
    )

    print()
    print(
        "UNIQUE VALUE ANALYSIS"
    )

    print(
        "-" * 80
    )

    print(
        f"Unique BF16 values : "
        f"{unique_count:,}"
    )

    print(
        f"Total weights      : "
        f"{parameter_count:,}"
    )

    print(
        f"Unique ratio       : "
        f"{(
            unique_count
            /
            parameter_count
        ):.8%}"
    )

    # --------------------------------------------------------
    # All experiment results.
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Rail count loop.
    # --------------------------------------------------------

    for rail_count in RAIL_COUNTS:

        print()
        print("=" * 80)
        print(
            f"RAILS = {rail_count}"
        )
        print("=" * 80)

        rail_start = (
            time.perf_counter()
        )

        rails = select_rails(
            unique_values,
            counts,
            rail_count
        )

        rail_seconds = (
            time.perf_counter()
            -
            rail_start
        )

        print(
            f"Rail selection time : "
            f"{rail_seconds:.4f}s"
        )

        print(
            f"Actual rails        : "
            f"{len(rails)}"
        )

        # ----------------------------------------------------
        # Max terms loop.
        # ----------------------------------------------------

        for max_terms in MAX_TERMS_LIST:

            print()
            print(
                "-" * 80
            )

            print(
                f"Rails={rail_count}, "
                f"MaxTerms={max_terms}"
            )

            compile_start = (
                time.perf_counter()
            )

            (
                route_table,
                exact_count
            ) = build_route_table_for_unique_values(
                unique_values,
                rails,
                max_terms
            )

            compile_seconds = (
                time.perf_counter()
                -
                compile_start
            )

            stats = route_statistics(
                route_table,
                exact_count
            )

            full_exact = (
                full_tensor_exact_check(
                    raw,
                    route_table
                )
            )

            # ------------------------------------------------
            # Only expand the 7.96M tensor if exact.
            # ------------------------------------------------

            expansion_seconds = 0.0

            if full_exact:

                expansion_start = (
                    time.perf_counter()
                )

                expanded_route_ids, route_list = (
                    expand_routes(
                        raw,
                        route_table
                    )
                )

                expansion_seconds = (
                    time.perf_counter()
                    -
                    expansion_start
                )

                expansion_ok = np.all(
                    expanded_route_ids
                    >=
                    0
                )

                del expanded_route_ids
                del route_list

            else:

                expansion_ok = False

            storage = representation_estimate(
                rail_count,
                route_table,
                parameter_count
            )

            print(
                f"Exact unique values : "
                f"{stats['exact_unique']:,}"
                f" / "
                f"{unique_count:,}"
            )

            print(
                f"Missing unique      : "
                f"{stats['missing_unique']:,}"
            )

            print(
                f"Exact ratio         : "
                f"{stats['exact_ratio']:.8%}"
            )

            print(
                f"Average terms       : "
                f"{stats['average_terms']:.4f}"
            )

            print(
                f"Max actual terms    : "
                f"{stats['max_terms']}"
            )

            print(
                f"Full tensor exact   : "
                f"{bool(full_exact)}"
            )

            print(
                f"Full route expand   : "
                f"{bool(expansion_ok)}"
            )

            print(
                f"Unique route bits   : "
                f"{storage['unique_route_bits']:,}"
            )

            print(
                f"Full tensor route bits: "
                f"{storage['full_tensor_route_bits']:,}"
            )

            print(
                f"Full total bits     : "
                f"{storage['full_total_bits']:,}"
            )

            print(
                f"Original bits       : "
                f"{storage['original_bits']:,}"
            )

            print(
                f"Unique compression  : "
                f"{storage['unique_compression']:.4f}x"
            )

            print(
                f"Full compression    : "
                f"{storage['full_compression']:.4f}x"
            )

            print(
                f"Compile time        : "
                f"{compile_seconds:.4f}s"
            )

            print(
                f"Expansion time      : "
                f"{expansion_seconds:.4f}s"
            )

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
                    stats[
                        "exact_unique"
                    ]
                ),

                "missing_unique": int(
                    stats[
                        "missing_unique"
                    ]
                ),

                "exact_ratio": float(
                    stats[
                        "exact_ratio"
                    ]
                ),

                "average_terms": float(
                    stats[
                        "average_terms"
                    ]
                ),

                "max_actual_terms": int(
                    stats[
                        "max_terms"
                    ]
                ),

                "full_tensor_exact": bool(
                    full_exact
                ),

                "full_route_expansion_ok": bool(
                    expansion_ok
                ),

                "rail_bits": int(
                    storage[
                        "rail_bits"
                    ]
                ),

                "route_index_bits": int(
                    storage[
                        "route_index_bits"
                    ]
                ),

                "unique_route_bits": int(
                    storage[
                        "unique_route_bits"
                    ]
                ),

                "unique_active_terms": int(
                    storage[
                        "unique_active_terms"
                    ]
                ),

                "unique_total_bits": int(
                    storage[
                        "unique_total_bits"
                    ]
                ),

                "full_route_id_bits": int(
                    storage[
                        "full_route_id_bits"
                    ]
                ),

                "full_tensor_route_bits": int(
                    storage[
                        "full_tensor_route_bits"
                    ]
                ),

                "full_total_bits": int(
                    storage[
                        "full_total_bits"
                    ]
                ),

                "original_bits": int(
                    storage[
                        "original_bits"
                    ]
                ),

                "unique_compression": float(
                    storage[
                        "unique_compression"
                    ]
                ),

                "full_compression": float(
                    storage[
                        "full_compression"
                    ]
                ),

                "compile_seconds": float(
                    compile_seconds
                ),

                "expansion_seconds": float(
                    expansion_seconds
                ),
            }

            results.append(
                result
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

    print(
        "rails,max_terms,"
        "exact_unique,missing_unique,"
        "exact_ratio,full_tensor_exact,"
        "unique_compression,full_compression,"
        "compile_seconds"
    )

    for result in results:

        print(
            f"{result['rails']},"
            f"{result['max_terms']},"
            f"{result['exact_unique']},"
            f"{result['missing_unique']},"
            f"{result['exact_ratio']:.8f},"
            f"{result['full_tensor_exact']},"
            f"{result['unique_compression']:.4f},"
            f"{result['full_compression']:.4f},"
            f"{result['compile_seconds']:.4f}"
        )

    # ========================================================
    # BEST
    # ========================================================

    exact_results = [
        result
        for result in results
        if result["full_tensor_exact"]
    ]

    print()
    print("=" * 80)
    print(
        "LOSSLESS CONFIGURATIONS"
    )
    print("=" * 80)

    if exact_results:

        exact_results.sort(
            key=lambda x: x[
                "full_compression"
            ],
            reverse=True
        )

        for result in exact_results:

            print(
                result
            )

    else:

        print(
            "No full-tensor lossless configuration found."
        )

    # ========================================================
    # JSON SAVE
    # ========================================================

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
        "results": results,
    }

    output = make_json_safe(
        output
    )

    with open(
        "railnet_bf16_compile_results.json",
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
        "SAVED"
    )
    print("=" * 80)

    print(
        "railnet_bf16_compile_results.json"
    )

    print()
    print("=" * 80)
    print(
        "COMPILATION EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()