# main.py
import struct
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd


# ============================================================
# RAILNET-256 GENERALIZATION BENCHMARK
#
# Goal:
#
#   Test whether a fixed shared-rail fabric can represent
#   increasingly diverse weight matrices exactly.
#
#   Unique weights:
#       16
#       32
#       64
#       128
#       256
#
#   Shared rails:
#       fixed = 16
#
# No per-weight floating coefficient.
# ============================================================


SEED = 42

INPUT_SIZE = 16
OUTPUT_SIZE = 16

PARAMETERS = (
    INPUT_SIZE *
    OUTPUT_SIZE
)

FIXED_RAIL_COUNT = 16

UNIQUE_LEVELS = [
    16,
    32,
    64,
    128,
    256,
]

OUTPUT_TESTS = 1000


# ============================================================
# FP32 BIT UTILITIES
# ============================================================

def float32_bits(x):
    return struct.unpack(
        "<I",
        struct.pack(
            "<f",
            float(np.float32(x))
        )
    )[0]


def float32_from_bits(bits):
    return np.float32(
        struct.unpack(
            "<f",
            struct.pack(
                "<I",
                int(bits) & 0xFFFFFFFF
            )
        )[0]
    )


def arrays_bitwise_equal(a, b):
    return np.array_equal(
        a.astype(np.float32).view(np.uint32),
        b.astype(np.float32).view(np.uint32),
    )


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Rail:

    rail_id: int
    value: np.float32
    bits: int


@dataclass(frozen=True)
class Route:

    rail_id: int


@dataclass
class Fabric:

    rails: tuple
    routes: tuple


# ============================================================
# MODEL GENERATION
# ============================================================

def create_model(unique_count, seed=SEED):
    """
    Create a 16x16 FP32 model containing exactly
    `unique_count` unique values.

    For 256:
        every weight is unique.

    For smaller values:
        weights are shared deliberately.
    """

    rng = np.random.default_rng(
        seed + unique_count
    )

    # Generate unique candidate float32 values.
    values = set()

    while len(values) < unique_count:

        candidates = rng.normal(
            0.0,
            0.5,
            size=unique_count * 2
        ).astype(np.float32)

        for value in candidates:

            values.add(
                float32_bits(value)
            )

            if len(values) >= unique_count:
                break

    bit_values = list(
        values
    )[:unique_count]

    dictionary = np.array(
        [
            float32_from_bits(x)
            for x in bit_values
        ],
        dtype=np.float32
    )

    # Shuffle assignment across matrix.
    assignments = rng.integers(
        0,
        unique_count,
        size=PARAMETERS
    )

    # Make sure every dictionary element appears at least once.
    if unique_count <= PARAMETERS:

        assignments[
            :unique_count
        ] = np.arange(
            unique_count
        )

    rng.shuffle(
        assignments
    )

    weights = dictionary[
        assignments
    ].reshape(
        INPUT_SIZE,
        OUTPUT_SIZE
    )

    return weights


# ============================================================
# RAIL COMPILATION
# ============================================================

def compile_shared_rails(
    weights,
    rail_count
):
    """
    Shared rails correspond to exact unique FP32 values
    only when the number of unique values <= rail_count.

    For larger models we deliberately use a deterministic
    nearest-rail approximation. This lets us discover the
    exact representability boundary.
    """

    flat = weights.reshape(-1)

    unique = {}

    for value in flat:

        bits = float32_bits(
            value
        )

        unique[
            bits
        ] = np.float32(value)

    unique_values = np.array(
        list(
            unique.values()
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Exact case:
    # --------------------------------------------------------

    if len(unique_values) <= rail_count:

        selected = unique_values

        # Pad with unused rails.
        if len(selected) < rail_count:

            padding = np.zeros(
                rail_count - len(selected),
                dtype=np.float32
            )

            selected = np.concatenate(
                [
                    selected,
                    padding
                ]
            )

        rails = tuple(
            Rail(
                rail_id=i,
                value=selected[i],
                bits=float32_bits(
                    selected[i]
                ),
            )
            for i in range(rail_count)
        )

        return rails

    # --------------------------------------------------------
    # Approximate case:
    #
    # Choose deterministic quantile basis.
    # --------------------------------------------------------

    values64 = np.sort(
        unique_values.astype(
            np.float64
        )
    )

    positions = np.linspace(
        0,
        len(values64) - 1,
        rail_count
    ).astype(int)

    selected = np.array(
        [
            values64[p]
            for p in positions
        ],
        dtype=np.float32
    )

    return tuple(
        Rail(
            rail_id=i,
            value=selected[i],
            bits=float32_bits(
                selected[i]
            ),
        )
        for i in range(rail_count)
    )


# ============================================================
# ROUTING
# ============================================================

def build_routes(
    weights,
    rails
):
    """
    Each weight gets exactly one rail.

    No coefficient.
    """

    rail_values = np.array(
        [
            float(r.value)
            for r in rails
        ],
        dtype=np.float64
    )

    routes = []

    for value in weights.reshape(-1):

        target = float(
            np.float32(value)
        )

        index = int(
            np.argmin(
                np.abs(
                    rail_values
                    -
                    target
                )
            )
        )

        routes.append(
            Route(
                rail_id=index
            )
        )

    return tuple(routes)


# ============================================================
# RECONSTRUCTION
# ============================================================

def reconstruct(
    fabric
):

    output = np.empty(
        PARAMETERS,
        dtype=np.float32
    )

    rail_values = np.array(
        [
            rail.value
            for rail in fabric.rails
        ],
        dtype=np.float32
    )

    for i, route in enumerate(
        fabric.routes
    ):

        output[i] = (
            rail_values[
                route.rail_id
            ]
        )

    return output.reshape(
        INPUT_SIZE,
        OUTPUT_SIZE
    )


# ============================================================
# SHARED MULTIPLICATION COUNT
# ============================================================

def shared_multiplier_count(
    fabric
):

    pairs = set()

    for flat_index, route in enumerate(
        fabric.routes
    ):

        output_index = (
            flat_index
            %
            OUTPUT_SIZE
        )

        pairs.add(
            (
                route.rail_id,
                output_index
            )
        )

    return len(
        pairs
    )


# ============================================================
# FANOUT
# ============================================================

def fanout_stats(
    fabric
):

    fanout = {
        rail.rail_id: 0
        for rail in fabric.rails
    }

    for route in fabric.routes:

        fanout[
            route.rail_id
        ] += 1

    values = np.array(
        list(
            fanout.values()
        )
    )

    active = (
        values[
            values > 0
        ]
    )

    return {
        "max_fanout":
            int(
                np.max(active)
            )
            if len(active)
            else 0,

        "average_fanout":
            float(
                np.mean(active)
            )
            if len(active)
            else 0.0,

        "active_rails":
            int(
                len(active)
            ),
    }


# ============================================================
# EXACT MATHEMATICAL ORACLE
# ============================================================

def exact_dense_oracle(
    x,
    weights
):

    outputs = []

    for j in range(
        OUTPUT_SIZE
    ):

        total = Fraction(
            0,
            1
        )

        for i in range(
            INPUT_SIZE
        ):

            xf = Fraction.from_float(
                float(
                    np.float32(
                        x[i]
                    )
                )
            )

            wf = Fraction.from_float(
                float(
                    np.float32(
                        weights[i, j]
                    )
                )
            )

            total += (
                xf * wf
            )

        outputs.append(
            np.float32(
                float(total)
            )
        )

    return np.array(
        outputs,
        dtype=np.float32
    )


# ============================================================
# EXACT RAIL ORACLE
# ============================================================

def exact_rail_oracle(
    x,
    fabric
):

    outputs = []

    rail_fraction = {
        rail.rail_id:
            Fraction.from_float(
                float(
                    rail.value
                )
            )
        for rail in fabric.rails
    }

    for j in range(
        OUTPUT_SIZE
    ):

        total = Fraction(
            0,
            1
        )

        for rail in fabric.rails:

            input_sum = Fraction(
                0,
                1
            )

            used = False

            for i in range(
                INPUT_SIZE
            ):

                route_index = (
                    i
                    *
                    OUTPUT_SIZE
                    +
                    j
                )

                route = fabric.routes[
                    route_index
                ]

                if (
                    route.rail_id
                    !=
                    rail.rail_id
                ):
                    continue

                used = True

                xf = Fraction.from_float(
                    float(
                        np.float32(
                            x[i]
                        )
                    )
                )

                input_sum += xf

            if used:

                total += (
                    input_sum
                    *
                    rail_fraction[
                        rail.rail_id
                    ]
                )

        outputs.append(
            np.float32(
                float(total)
            )
        )

    return np.array(
        outputs,
        dtype=np.float32
    )


# ============================================================
# EXECUTION TEST
# ============================================================

def execution_test(
    weights,
    fabric,
    tests=OUTPUT_TESTS
):

    rng = np.random.default_rng(
        SEED + 5000
    )

    oracle_failures = 0
    fp32_failures = 0

    maximum_error = 0.0

    for _ in range(
        tests
    ):

        x = rng.normal(
            0.0,
            1.0,
            INPUT_SIZE
        ).astype(np.float32)

        dense = exact_dense_oracle(
            x,
            weights
        )

        rail = exact_rail_oracle(
            x,
            fabric
        )

        if not np.array_equal(
            dense,
            rail
        ):
            oracle_failures += 1

        # Normal FP32 implementation only
        # for diagnostic purposes.
        dense_fp32 = (
            x @ weights
        ).astype(np.float32)

        rail_fp32 = (
            normal_rail_forward(
                x,
                fabric
            )
        )

        if not np.array_equal(
            dense_fp32,
            rail_fp32
        ):
            fp32_failures += 1

        current_error = np.max(
            np.abs(
                dense.astype(
                    np.float64
                )
                -
                rail.astype(
                    np.float64
                )
            )
        )

        maximum_error = max(
            maximum_error,
            float(current_error)
        )

    return {
        "oracle_failures":
            oracle_failures,

        "fp32_failures":
            fp32_failures,

        "oracle_exact":
            oracle_failures == 0,

        "max_oracle_error":
            maximum_error,
    }


# ============================================================
# NORMAL FP32 RAIL FORWARD
# ============================================================

def normal_rail_forward(
    x,
    fabric
):

    output = np.zeros(
        OUTPUT_SIZE,
        dtype=np.float32
    )

    for rail in fabric.rails:

        for j in range(
            OUTPUT_SIZE
        ):

            total = np.float32(
                0.0
            )

            active = False

            for i in range(
                INPUT_SIZE
            ):

                route_index = (
                    i
                    *
                    OUTPUT_SIZE
                    +
                    j
                )

                route = fabric.routes[
                    route_index
                ]

                if (
                    route.rail_id
                    !=
                    rail.rail_id
                ):
                    continue

                active = True

                total = np.float32(
                    total
                    +
                    x[i]
                )

            if active:

                output[j] = np.float32(
                    output[j]
                    +
                    (
                        total
                        *
                        rail.value
                    )
                )

    return output


# ============================================================
# METRICS
# ============================================================

def analyze(
    weights,
    fabric
):

    reconstructed = reconstruct(
        fabric
    )

    weight_exact = arrays_bitwise_equal(
        weights,
        reconstructed
    )

    different = np.count_nonzero(
        weights.astype(np.float32).view(np.uint32)
        !=
        reconstructed.astype(np.float32).view(np.uint32)
    )

    diff = (
        weights.astype(np.float64)
        -
        reconstructed.astype(np.float64)
    )

    multiplications = (
        shared_multiplier_count(
            fabric
        )
    )

    fanout = fanout_stats(
        fabric
    )

    rail_bits = (
        FIXED_RAIL_COUNT
        *
        32
    )

    route_bits_per_weight = int(
        np.ceil(
            np.log2(
                FIXED_RAIL_COUNT
            )
        )
    )

    route_bits = (
        PARAMETERS
        *
        route_bits_per_weight
    )

    total_bits = (
        rail_bits
        +
        route_bits
    )

    original_bits = (
        PARAMETERS
        *
        32
    )

    return {
        "unique_weights":
            len(
                set(
                    weights.reshape(-1)
                    .astype(np.float32)
                    .view(np.uint32)
                )
            ),

        "rails":
            FIXED_RAIL_COUNT,

        "weight_exact":
            weight_exact,

        "different_weights":
            int(different),

        "weight_rmse":
            float(
                np.sqrt(
                    np.mean(
                        diff ** 2
                    )
                )
            ),

        "max_weight_error":
            float(
                np.max(
                    np.abs(diff)
                )
            ),

        "shared_multiplications":
            multiplications,

        "direct_multiplications":
            PARAMETERS,

        "multiplier_reduction":
            (
                1.0
                -
                multiplications
                /
                PARAMETERS
            ),

        "active_rails":
            fanout["active_rails"],

        "max_fanout":
            fanout["max_fanout"],

        "average_fanout":
            fanout["average_fanout"],

        "rail_bits":
            rail_bits,

        "routing_bits":
            route_bits,

        "representation_bits":
            total_bits,

        "original_bits":
            original_bits,

        "compression":
            (
                original_bits
                /
                total_bits
            ),
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 80)
    print(
        "RAILNET-256 GENERALIZATION BENCHMARK"
    )
    print("=" * 80)

    print()
    print(
        f"Parameters       : {PARAMETERS}"
    )

    print(
        f"Shared rails     : {FIXED_RAIL_COUNT}"
    )

    print(
        f"Unique levels    : {UNIQUE_LEVELS}"
    )

    print()

    results = []

    for unique_count in UNIQUE_LEVELS:

        print()
        print("-" * 80)

        print(
            f"UNIQUE WEIGHTS = {unique_count}"
        )

        print("-" * 80)

        start = time.perf_counter()

        weights = create_model(
            unique_count
        )

        compile_start = (
            time.perf_counter()
        )

        rails = compile_shared_rails(
            weights,
            FIXED_RAIL_COUNT
        )

        routes = build_routes(
            weights,
            rails
        )

        fabric = Fabric(
            rails=rails,
            routes=routes
        )

        compile_time = (
            time.perf_counter()
            -
            compile_start
        )

        result = analyze(
            weights,
            fabric
        )

        execution = execution_test(
            weights,
            fabric
        )

        total_time = (
            time.perf_counter()
            -
            start
        )

        result.update(
            {
                "oracle_exact":
                    execution[
                        "oracle_exact"
                    ],

                "oracle_failures":
                    execution[
                        "oracle_failures"
                    ],

                "fp32_failures":
                    execution[
                        "fp32_failures"
                    ],

                "max_oracle_error":
                    execution[
                        "max_oracle_error"
                    ],

                "compile_seconds":
                    compile_time,

                "total_seconds":
                    total_time,
            }
        )

        results.append(
            result
        )

        print()
        print(
            f"Unique weights       : "
            f"{result['unique_weights']}"
        )

        print(
            f"Shared rails        : "
            f"{result['rails']}"
        )

        print(
            f"Exact weights       : "
            f"{result['weight_exact']}"
        )

        print(
            f"Weight RMSE         : "
            f"{result['weight_rmse']:.12e}"
        )

        print(
            f"Shared multipliers  : "
            f"{result['shared_multiplications']}"
        )

        print(
            f"Multiplier reduction: "
            f"{result['multiplier_reduction'] * 100:.2f}%"
        )

        print(
            f"Max fanout          : "
            f"{result['max_fanout']}"
        )

        print(
            f"Active rails        : "
            f"{result['active_rails']}"
        )

        print(
            f"Representation bits : "
            f"{result['representation_bits']}"
        )

        print(
            f"Compression         : "
            f"{result['compression']:.4f}x"
        )

        print(
            f"Oracle exact        : "
            f"{result['oracle_exact']}"
        )

        print(
            f"Oracle failures     : "
            f"{result['oracle_failures']}"
        )

        print(
            f"FP32 diagnostic fail: "
            f"{result['fp32_failures']}"
        )

        print(
            f"Compile time        : "
            f"{result['compile_seconds']:.6f}s"
        )

        print(
            f"Total time          : "
            f"{result['total_seconds']:.6f}s"
        )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    df = pd.DataFrame(
        results
    )

    print()
    print("=" * 80)
    print(
        "FINAL TABLE"
    )
    print("=" * 80)

    print(
        df[
            [
                "unique_weights",
                "rails",
                "weight_exact",
                "weight_rmse",
                "shared_multiplications",
                "multiplier_reduction",
                "active_rails",
                "max_fanout",
                "representation_bits",
                "compression",
                "oracle_exact",
                "oracle_failures",
                "fp32_failures",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        "railnet_256_generalization.csv",
        index=False
    )

    # ========================================================
    # IMPORTANT FINDINGS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "INTERPRETATION"
    )
    print("=" * 80)

    exact = df[
        df["weight_exact"]
        &
        df["oracle_exact"]
    ]

    if len(exact):

        print()
        print(
            "Exact configurations:"
        )

        print(
            exact[
                [
                    "unique_weights",
                    "rails",
                    "shared_multiplications",
                    "multiplier_reduction",
                    "compression"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "No exact generalized configuration found."
        )

    print()
    print(
        "Next boundary to inspect:"
    )

    print(
        "16 → 32 → 64 → 128 → 256 unique FP32 weights."
    )

    print()
    print(
        "The critical question is:"
    )

    print(
        "Can 16 shared rails remain exact as weight "
        "diversity increases?"
    )

    print()
    print(
        "Saved:"
    )

    print(
        "railnet_256_generalization.csv"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()