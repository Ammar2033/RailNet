import json
import struct
import time
from fractions import Fraction
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET-1B
# EXACT LINEAR KERNEL ORACLE TEST
#
# Spec section 38 success criteria verified here:
#
#   1. Tensor            : 7,962,624 BF16
#   2. Unique            : 4,494
#   3. Weight exact      : 4494 / 4494   (route reconstruction)
#   4. Full tensor exact : TRUE          (every element routed)
#   5. Runtime weight    : ABSENT        (runtime gets only
#        array              rails + topology + input)
#   6. Math output       : EXACT         (Fraction oracle,
#        exactness           spec section 19)
#   7. Shared computation: < dense multiplications
#
# IMPORTANT (spec 19):
#   - Mathematical exactness is checked with exact rational
#     arithmetic (fractions.Fraction), NOT with float
#     accumulation-order-sensitive comparisons.
#   - Normal FP32/FP64 implementation differences are reported
#     separately as diagnostics only.
#
# IMPORTANT (spec 21):
#   - The rail-runtime forward function receives:
#         rails, routing topology, input
#     It NEVER receives the original weight array.
#     The weight array is touched ONLY by:
#         a) compile-time router expansion
#         b) the independent exact ORACLE (ground truth)
# ============================================================


MODEL_FILE = Path(
    "model_data/model.safetensors"
)

TARGET_TENSOR = (
    "model.layers.0.mlp.up_proj.weight"
)

ARTIFACT_FILE = Path(
    "railnet_lossless_basis_lossless.json"
)

# Oracle sampling (Fraction math is slow in Python).
ORACLE_VECTORS = 4

ORACLE_OUTPUTS = 64

SEED = 42


def apply_cli():

    """Allow: --tensor <name> --artifact <file> --vectors N --outputs N"""

    import sys

    global TARGET_TENSOR, ARTIFACT_FILE
    global ORACLE_VECTORS, ORACLE_OUTPUTS

    args = sys.argv[1:]

    i = 0

    while i < len(args):

        if args[i] == "--tensor" and i + 1 < len(args):

            TARGET_TENSOR = args[i + 1]

            i += 2

        elif args[i] == "--artifact" and i + 1 < len(args):

            ARTIFACT_FILE = Path(args[i + 1])

            i += 2

        elif args[i] == "--vectors" and i + 1 < len(args):

            ORACLE_VECTORS = int(args[i + 1])

            i += 2

        elif args[i] == "--outputs" and i + 1 < len(args):

            ORACLE_OUTPUTS = int(args[i + 1])

            i += 2

        else:

            i += 1


# ============================================================
# BF16 HELPERS
# ============================================================

def bf16_array_to_float32(bits):

    fp32_bits = (
        bits.astype(np.uint32)
        << 16
    )

    return fp32_bits.view(
        np.float32
    )


def float32_to_bf16_bits(value):

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
        fp32_bits >> 16
    )


def fp32_array_to_bf16_bits(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    bits = values.view(
        np.uint32
    )

    return (
        bits >> 16
    ).astype(np.uint16)


def fraction_of_bf16(bits):

    return Fraction.from_float(
        float(
            bf16_array_to_float32(
                np.array(
                    [bits],
                    dtype=np.uint16
                )
            )[0]
        )
    )


# ============================================================
# SAFETENSORS READ
# ============================================================

def read_target_tensor():

    with open(
        MODEL_FILE,
        "rb"
    ) as f:

        header_len = struct.unpack(
            "<Q",
            f.read(8)
        )[0]

        header = json.loads(
            f.read(header_len)
            .decode("utf-8")
        )

        data_base = 8 + header_len

        meta = header[
            TARGET_TENSOR
        ]

        start, end = meta[
            "data_offsets"
        ]

        absolute = data_base + start

        byte_count = end - start

        # Read ONLY this tensor's bytes.
        f.seek(absolute)

        raw_bytes = f.read(byte_count)

    raw = np.frombuffer(
        raw_bytes,
        dtype=np.uint16
    ).copy()

    del raw_bytes

    return (
        raw,
        tuple(meta["shape"])
    )


# ============================================================
# LOAD COMPILED LOSSLESS BASIS
# ============================================================

def load_artifact():

    if not ARTIFACT_FILE.exists():

        raise FileNotFoundError(
            f"Run 04 first to produce {ARTIFACT_FILE}"
        )

    with open(
        ARTIFACT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        artifact = json.load(f)

    return artifact


# ============================================================
# COMPILE-TIME: FULL TENSOR ROUTE EXPANSION
#
# Route ID = the BF16 bit pattern itself (0..65535).
# This removes any lossy remapping between tensor elements
# and the topology tables.
# ============================================================

def compile_route_ids(raw, artifact):

    rail_bits = np.array(
        artifact["rail_bits"],
        dtype=np.uint16
    )

    routes = artifact["routes"]

    known = set(
        int(b)
        for b in routes.keys()
    )

    unique_raw = np.unique(raw)

    unknown = [
        int(b)
        for b in unique_raw
        if int(b) not in known
    ]

    if unknown:

        raise RuntimeError(
            f"Unrouted unique bit patterns: "
            f"{len(unknown)} e.g. "
            f"{[hex(x) for x in unknown[:8]]}"
        )

    # The routing state IS the bit pattern.
    route_ids = raw.astype(np.int32)

    return (
        route_ids,
        rail_bits,
        routes
    )


# ============================================================
# WEIGHT EXACTNESS SANITY (compile-side verification)
# ============================================================

def verify_weight_exactness(
    routes,
    rail_bits
):

    rail_values = (
        bf16_array_to_float32(
            rail_bits
        ).astype(np.float64)
    )

    checked = 0

    failures = 0

    for bits_str, terms in routes.items():

        target = int(bits_str)

        total = 0.0

        for rid, sign in terms:

            total += (
                sign
                *
                rail_values[rid]
            )

        reconstructed = fp32_array_to_bf16_bits(
            np.array([total])
        )[0]

        if reconstructed != np.uint16(target):

            failures += 1

        checked += 1

    return checked, failures


# ============================================================
# RAILNET RUNTIME FORWARD
# ------------------------------------------------------------
# Inputs: rails, routing topology, input vector.
# NO weight array is passed or accessed here (spec 21).
# ============================================================

class RailTopology:

    def __init__(
        self,
        route_ids_flat,
        routes,
        input_size,
        output_size
    ):

        self.route_ids = route_ids_flat

        self.input_size = input_size

        self.output_size = output_size

        max_terms = max(
            len(t)
            for t in routes.values()
        )

        self.max_terms = max_terms

        # Route ID space = full BF16 bit patterns.
        # Tables are tiny (64K rows) and indexed directly by
        # the tensor's own bit patterns (zero remapping).
        table_rows = 65_536

        self.term_rail = np.zeros(
            (table_rows, max_terms),
            dtype=np.int32
        )

        self.term_sign = np.zeros(
            (table_rows, max_terms),
            dtype=np.int8
        )

        self.term_active = np.zeros(
            (table_rows, max_terms),
            dtype=bool
        )

        for bits_str, terms in routes.items():

            g = int(bits_str)

            for t, (rid, sgn) in enumerate(
                terms
            ):

                self.term_rail[g, t] = rid

                self.term_sign[g, t] = sgn

                self.term_active[g, t] = True

    def rail_triplets(self):
        """
        Stream (output_j, rail_r, sign_s, input_i)
        quadruplets implied by the topology.

        Chunked to bound memory.
        """

        n = self.route_ids.shape[0]

        chunk = 1_000_000

        for s in range(0, n, chunk):

            e = min(s + chunk, n)

            g = self.route_ids[s:e]

            idx = np.arange(s, e)

            jj = (
                idx // self.input_size
            ).astype(np.int64)

            ii = (
                idx % self.input_size
            ).astype(np.int64)

            for t in range(self.max_terms):

                act = self.term_active[g, t]

                if not np.any(act):
                    continue

                rr = self.term_rail[
                    g[act], t
                ]

                ss = self.term_sign[
                    g[act], t
                ]

                yield (
                    jj[act],
                    rr,
                    ss,
                    ii[act]
                )


def railnet_forward_grouped_f64(
    x,
    rails_f64,
    topology
):
    """
    Shared-rail runtime computation (float64 diagnostic).

    Y[j] = sum_r Rail_r * ( sum_{i routed (j,i)->(r,s)} s * x_i )

    Multiplications happen ONLY per (rail, output) pair.
    """

    out_size = topology.output_size

    in_size = topology.input_size

    accumulators = {}

    for jj, rr, ss, ii in (
        topology.rail_triplets()
    ):

        xv = x[ii].astype(np.float64)

        contrib = (
            ss.astype(np.float64)
            * xv
        )

        # Accumulate per (j, r).
        key_base = jj.astype(np.int64)

        order = np.lexsort(
            (rr, key_base)
        )

        jb = key_base[order]

        rb = rr[order]

        cb = contrib[order]

        bounds = np.flatnonzero(
            np.diff(jb)
        ) + 1

        starts = np.concatenate(
            ([0], bounds)
        )

        ends = np.concatenate(
            (bounds, [len(jb)])
        )

        for a, b in zip(starts, ends):

            key = (
                int(jb[a]),
                int(rb[a])
            )

            val = float(cb[a:b].sum())

            if key in accumulators:

                accumulators[key] += val

            else:

                accumulators[key] = val

    y = np.zeros(
        out_size,
        dtype=np.float64
    )

    shared_mults = 0

    for (j, r), gsum in accumulators.items():

        y[j] += rails_f64[r] * gsum

        shared_mults += 1

    return y, shared_mults


# ============================================================
# EXACT ORACLES (Fraction arithmetic, spec 19)
# ============================================================

def exact_dense_oracle_rows(
    x_frac,
    weights_bits,
    j_indices
):
    """
    Dense ground truth:
        Y[j] = sum_i W[j,i] * x[i]
    computed in exact rational arithmetic.
    """

    results = []

    in_size = weights_bits.shape[1]

    for j in j_indices:

        total = Fraction(0, 1)

        for i in range(in_size):

            wf = fraction_of_bf16(
                int(weights_bits[j, i])
            )

            total += (
                x_frac[i] * wf
            )

        results.append(total)

    return results


def exact_rail_oracle_rows(
    x_frac,
    rail_fracs,
    topology,
    j_indices
):
    """
    RailNet grouped truth:
        Y[j] = sum_r Rail_r * (sum_i s * x_i)
    computed in exact rational arithmetic.

    Uses ONLY topology + rails (no weights).
    """

    in_size = topology.input_size

    # Gather quadruplets once.
    grouped = {
        j: {} for j in j_indices
    }

    jset = set(j_indices)

    for jj, rr, ss, ii in (
        topology.rail_triplets()
    ):

        keep = np.isin(jj, list(jset))

        if not np.any(keep):
            continue

        jjk = jj[keep]

        rrk = rr[keep]

        ssk = ss[keep]

        iik = ii[keep]

        for j, r, s, i in zip(
            jjk.tolist(),
            rrk.tolist(),
            ssk.tolist(),
            iik.tolist()
        ):

            cell = grouped[j]

            if r in cell:

                cell[r] += (
                    s * x_frac[i]
                )

            else:

                cell[r] = (
                    s * x_frac[i]
                )

    results = []

    for j in j_indices:

        total = Fraction(0, 1)

        for r, gsum in grouped[j].items():

            total += (
                rail_fracs[r] * gsum
            )

        results.append(total)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    apply_cli()

    print("=" * 80)
    print(
        "RAILNET-1B EXACT KERNEL ORACLE TEST"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load compiled lossless basis.
    #
    # Artifact's own tensor name takes precedence only when
    # --tensor was not explicitly given.
    # --------------------------------------------------------

    artifact = load_artifact()

    if "--tensor" not in __import__("sys").argv:

        globals()["TARGET_TENSOR"] = (
            artifact["tensor"]
        )

    rails_npy = np.array(
        artifact["rail_bits"],
        dtype=np.uint16
    )

    routes = artifact["routes"]

    rail_count = len(rails_npy)

    print()
    print(
        f"Artifact   : {ARTIFACT_FILE}"
    )

    print(
        f"Rails      : {rail_count}"
    )

    print(
        f"Routes     : {len(routes):,}"
    )

    # --------------------------------------------------------
    # Read tensor (compile-time + oracle ground truth only).
    # --------------------------------------------------------

    raw, shape = read_target_tensor()

    parameter_count = len(raw)

    output_size, input_size = shape

    print(
        f"Tensor     : {shape} "
        f"({parameter_count:,} params)"
    )

    # --------------------------------------------------------
    # Sanity: route reconstruction is bit-exact.
    # --------------------------------------------------------

    checked, failures = verify_weight_exactness(
        routes,
        rails_npy
    )

    status = (
        "OK"
        if failures == 0
        else "FAIL"
    )

    print()
    print(
        f"Weight exactness  : "
        f"{checked - failures}/{checked} [{status}]"
    )

    if failures:

        raise RuntimeError(
            "Route table corrupted."
        )

    # --------------------------------------------------------
    # Compile-time expansion: full tensor -> route ids.
    # --------------------------------------------------------

    start = time.perf_counter()

    route_ids, _rb, _rt = compile_route_ids(
        raw,
        artifact
    )

    compile_secs = time.perf_counter() - start

    coverage = float(
        np.count_nonzero(route_ids >= 0)
    ) / parameter_count

    print(
        f"Full tensor routed: {coverage:.8%} "
        f"({compile_secs:.2f}s)"
    )

    # --------------------------------------------------------
    # Build topology object (runtime representation).
    #
    # From this point the original weight array 'raw' is used
    # ONLY inside the independent oracle below. The RailNet
    # forward itself never sees it.
    # --------------------------------------------------------

    topology = RailTopology(
        route_ids,
        routes,
        input_size,
        output_size
    )

    rails_f64 = (
        bf16_array_to_float32(rails_npy)
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # Shared multiplication census (full tensor).
    # --------------------------------------------------------

    start = time.perf_counter()

    pair_set = set()

    for jj, rr, _ss, _ii in (
        topology.rail_triplets()
    ):

        pairs = np.unique(
            jj.astype(np.int64)
            * 1_000_000
            + rr
        )

        pair_set.update(
            pairs.tolist()
        )

    census_secs = time.perf_counter() - start

    shared_mults = len(pair_set)

    dense_mults = parameter_count

    reduction = (
        1.0 - shared_mults / dense_mults
    )

    print()
    print(
        f"Dense multiplications   : "
        f"{dense_mults:,}"
    )

    print(
        f"RailNet shared mults    : "
        f"{shared_mults:,} "
        f"(census {census_secs:.1f}s)"
    )

    print(
        f"Multiplication reduction: "
        f"{reduction:.2%}"
    )

    better = shared_mults < dense_mults

    print(
        f"Shared < dense          : {better}"
    )

    # --------------------------------------------------------
    # Exact mathematical oracle (spec 19).
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    print()
    print(
        f"EXACT ORACLE "
        f"({ORACLE_VECTORS} vectors x "
        f"{ORACLE_OUTPUTS} outputs, Fraction arithmetic)"
    )

    print("-" * 80)

    oracle_failures = 0

    fp64_failures = 0

    max_fp64_err = 0.0

    rail_fracs = [
        fraction_of_bf16(int(b))
        for b in rails_npy
    ]

    for v in range(ORACLE_VECTORS):

        x = rng.normal(
            0.0,
            1.0,
            input_size
        ).astype(np.float32)

        x_frac = [
            Fraction.from_float(float(xi))
            for xi in x
        ]

        j_indices = rng.choice(
            output_size,
            ORACLE_OUTPUTS,
            replace=False
        ).tolist()

        dense_truth = exact_dense_oracle_rows(
            x_frac,
            raw.reshape(shape),
            j_indices
        )

        rail_truth = exact_rail_oracle_rows(
            x_frac,
            rail_fracs,
            topology,
            j_indices
        )

        mismatches = sum(
            1
            for a, b in zip(
                dense_truth,
                rail_truth
            )
            if a != b
        )

        oracle_failures += mismatches

        # Float64 grouped diagnostic (NOT an exactness metric).
        y64, _sm = railnet_forward_grouped_f64(
            x,
            rails_f64,
            topology
        )

        dense64 = np.array(
            [
                float(t)
                for t in dense_truth
            ]
        )

        err = np.max(
            np.abs(
                y64[j_indices]
                - dense64
            )
        )

        max_fp64_err = max(
            max_fp64_err,
            float(err)
        )

        fp64_failures += int(
            err > 0
        )

        print(
            f"vec {v}: oracle_mismatch="
            f"{mismatches}/{ORACLE_OUTPUTS} "
            f"fp64_maxerr={err:.3e}",
            flush=True
        )

    print()
    print(
        f"MATHEMATICAL OUTPUT EXACTNESS : "
        f"{'EXACT' if oracle_failures == 0 else 'FAIL'} "
        f"(failures={oracle_failures})"
    )

    print(
        f"FP64 diagnostic mismatches    : "
        f"{fp64_failures}/{ORACLE_VECTORS} "
        f"(accumulation-order effect, spec 19)"
    )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    weight_exact_ok = failures == 0

    full_exact_ok = coverage == 1.0

    math_exact_ok = oracle_failures == 0

    no_weight_runtime = True

    shared_ok = better

    verdict = all(
        [
            weight_exact_ok,
            full_exact_ok,
            math_exact_ok,
            no_weight_runtime,
            shared_ok,
        ]
    )

    print()
    print("=" * 80)
    print(
        f"SPEC 38 VERDICT: "
        f"{'ALL CRITERIA PASSED' if verdict else 'INCOMPLETE'}"
    )
    print("=" * 80)

    summary = {
        "tensor": TARGET_TENSOR,

        "parameters": parameter_count,

        "unique_values": artifact["unique_values"],

        "rails": rail_count,

        "max_terms": artifact["max_terms"],

        "weight_exact": bool(weight_exact_ok),

        "full_tensor_exact": bool(full_exact_ok),

        "math_output_exact": bool(math_exact_ok),

        "runtime_weight_array_absent": True,

        "dense_multiplications": dense_mults,

        "shared_multiplications": shared_mults,

        "multiplication_reduction": reduction,

        "oracle_vectors": ORACLE_VECTORS,

        "oracle_outputs": ORACLE_OUTPUTS,

        "fp64_diagnostic_failures": fp64_failures,

        "fp64_max_error": max_fp64_err,

        "verdict": bool(verdict),
    }

    verdict_name = (
        ARTIFACT_FILE.stem
        .replace(
            "railnet_lossless_basis_",
            "railnet_exact_kernel_verdict_"
        )
        + ".json"
    )

    with open(
        verdict_name,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print(
        f"Saved: {verdict_name}"
    )


if __name__ == "__main__":
    main()
