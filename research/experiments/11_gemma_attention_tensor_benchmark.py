import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 11
# GEMMA LAYER-0 ATTENTION TENSORS - LOSSLESS BENCHMARK
#
# Reuses the proven MLP learned-basis pipeline (04):
#   uniform-rank init -> weighted coordinate descent
#   -> duplicate repair -> missing-value repair
#   -> safe-slot repair -> exhaustive exact evaluation
#
# Per tensor (q/k/v/o):
#   - find MINIMUM lossless (rails, terms) by ascending scan
#   - serialize compiled/layer0/<name>_lossless.json
#     (NO dense weights inside artifact, SHA-256 checksummed)
#   - verify Spec-16 pass criteria incl. Fraction oracle
#
# Pass criteria (spec 16):
#   unique exact = 100%
#   full tensor routed = TRUE
#   mathematical oracle = TRUE
#   runtime dense weight array = ABSENT
# ============================================================


MODEL_DIR = Path("model_data")

COMPILED_DIR = Path("compiled") / "layer0"

RESULTS_DIR = Path("results")

ATTENTION_TENSORS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
]

SHORT_NAMES = {
    "model.layers.0.self_attn.q_proj.weight": "q_proj",
    "model.layers.0.self_attn.k_proj.weight": "k_proj",
    "model.layers.0.self_attn.v_proj.weight": "v_proj",
    "model.layers.0.self_attn.o_proj.weight": "o_proj",
}

# Ascending rail scan for minimum lossless point.
RAIL_SCAN = [32, 64, 96, 128]

TERMS_PRIMARY = 4

TERMS_SECONDARY = 3

# Fraction-oracle sampling (validation tool, not runtime).
ORACLE_VECTORS = 2

ORACLE_OUTPUTS = 24

SEED = 42


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


HERE = Path(__file__).resolve().parent

RN = load_module(
    HERE / "04_bf16_learned_basis.py",
    "rn_compiler"
)

ORACLE = load_module(
    HERE / "05_bf16_exact_kernel_oracle.py",
    "rn_oracle"
)


# ============================================================
# ARTIFACT SERIALIZATION (spec 39 / 40)
# ============================================================

def build_artifact(
    tensor_name,
    shape,
    rail_bits,
    max_terms,
    route_table,
    exact_unique,
    unique_values,
    extra_validation
):

    routes_json = {
        str(bits): [
            [int(rid), int(sgn)]
            for rid, sgn in route
        ]
        for bits, route in sorted(
            route_table.items()
        )
    }

    content = {
        "magic": "RNET",

        "version": 1,

        "dtype": "BF16",

        "tensor": tensor_name,

        "shape": [
            int(x) for x in shape
        ],

        "parameters": int(
            np.prod(shape)
        ),

        "rail_count": int(len(rail_bits)),

        "max_terms": int(max_terms),

        "rails": [
            int(b) for b in rail_bits
        ],

        "routes": routes_json,

        "validation": {
            "unique_values": int(unique_values),

            "exact_unique": int(exact_unique),

            "exact_ratio": float(
                exact_unique / unique_values
                if unique_values else 0.0
            ),

            "weight_reconstruction": (
                extra_validation.get(
                    "reconstruction"
                )
            ),

            "full_tensor_routed": bool(
                extra_validation.get(
                    "full_routed"
                )
            ),

            "math_oracle_exact": bool(
                extra_validation.get(
                    "oracle"
                )
            ),

            "runtime_weight_array": "ABSENT",

            "seed": SEED,
        },
    }

    canonical = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")

    checksum = hashlib.sha256(
        canonical
    ).hexdigest()

    content["checksum_sha256"] = checksum

    return content


def verify_artifact_checksum(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    stored = data.pop(
        "checksum_sha256"
    )

    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")

    ok = (
        hashlib.sha256(canonical).hexdigest()
        == stored
    )

    return ok, len(data["routes"])


# ============================================================
# SHARED MULTIPLICATION CENSUS (bit-indexed topology)
# ============================================================

def census_shared_mults(
    raw,
    route_table,
    max_terms,
    output_size,
    input_size
):
    """
    Dense: every weight is one multiplication.
    RailNet: multiplications happen only per used
    (output_j, rail_r) pair (spec 36/37).
    """

    dense = raw.size

    term_rail = np.zeros(
        (65_536, max_terms),
        dtype=np.int32
    )

    term_active = np.zeros(
        (65_536, max_terms),
        dtype=bool
    )

    for bits_str, terms in route_table.items():

        g = int(bits_str)

        for t, (rid, _s) in enumerate(terms):

            term_rail[g, t] = rid

            term_active[g, t] = True

    pairs = set()

    n = raw.size

    chunk = 1_000_000

    for s in range(0, n, chunk):

        e = min(s + chunk, n)

        g = raw[s:e].astype(np.int32)

        idx = np.arange(s, e, dtype=np.int64)

        jj = idx // input_size

        for t in range(max_terms):

            act = term_active[g, t]

            if not np.any(act):

                continue

            rr = term_rail[g[act], t]

            pairs.update(
                (
                    jj[act].astype(np.int64)
                    * 70_000
                    + rr
                ).tolist()
            )

    return dense, len(pairs)


# ============================================================
# PER-TENSOR PIPELINE
# ============================================================

def process_tensor(tensor_name):

    short = SHORT_NAMES[tensor_name]

    print("=" * 80)
    print(f"TENSOR: {tensor_name}")
    print("=" * 80, flush=True)

    RN.TARGET_TENSOR = tensor_name

    raw, shape = RN.read_target_tensor()

    bits, counts, vals = (
        RN.analyze_unique_values(raw)
    )

    n_uniq = len(bits)

    print(
        f"shape={shape} params={len(raw):,} "
        f"unique={n_uniq:,}",
        flush=True
    )

    # --------------------------------------------------------
    # Ascending rail scan at TERMS_PRIMARY.
    # First lossless rails value wins (minimum rails).
    # --------------------------------------------------------

    found = None

    scan_log = []

    for rc in RAIL_SCAN:

        start = time.perf_counter()

        learned = RN.learn_basis(
            vals,
            bits,
            counts,
            rc,
            TERMS_PRIMARY
        )

        elapsed = time.perf_counter() - start

        table = RN.compile_exact_routes_exhaustive(
            bits,
            learned["rails"],
            TERMS_PRIMARY
        )

        cov = sum(
            1
            for b in bits
            if int(b) in table
        )

        lossless = cov == n_uniq

        scan_log.append(
            {
                "rails": rc,
                "terms": TERMS_PRIMARY,
                "exact": int(cov),
                "lossless": bool(lossless),
                "seconds": round(elapsed, 1),
            }
        )

        print(
            f"  [scan] rails={rc:3d}/{TERMS_PRIMARY}"
            f" exact={cov}/{n_uniq}"
            f" ({cov/n_uniq:.2%})"
            f" {'LOSSLESS' if lossless else ''}"
            f" [{elapsed:.0f}s]",
            flush=True
        )

        if lossless:

            found = {
                "rails": rc,
                "terms": TERMS_PRIMARY,
                "rails_arr": learned["rails"],
                "table": table,
            }

            break

    if found is None:

        print(
            "  no lossless config in scan grid",
            flush=True
        )

        return None

    # --------------------------------------------------------
    # Terms reduction at the minimal rail count.
    # Fewer terms -> fewer routing bits per edge.
    # --------------------------------------------------------

    start = time.perf_counter()

    learned3 = RN.learn_basis(
        vals,
        bits,
        counts,
        found["rails"],
        TERMS_SECONDARY
    )

    elapsed = time.perf_counter() - start

    table3 = RN.compile_exact_routes_exhaustive(
        bits,
        learned3["rails"],
        TERMS_SECONDARY
    )

    cov3 = sum(
        1
        for b in bits
        if int(b) in table3
    )

    lossless3 = cov3 == n_uniq

    scan_log.append(
        {
            "rails": found["rails"],
            "terms": TERMS_SECONDARY,
            "exact": int(cov3),
            "lossless": bool(lossless3),
            "seconds": round(elapsed, 1),
        }
    )

    print(
        f"  [scan] rails={found['rails']:3d}/"
        f"{TERMS_SECONDARY} exact={cov3}/{n_uniq}"
        f" ({cov3/n_uniq:.2%})"
        f" {'LOSSLESS' if lossless3 else ''}"
        f" [{elapsed:.0f}s]",
        flush=True
    )

    if lossless3:

        found = {
            "rails": found["rails"],
            "terms": TERMS_SECONDARY,
            "rails_arr": learned3["rails"],
            "table": table3,
        }

    # --------------------------------------------------------
    # Final validation (spec 16).
    # --------------------------------------------------------

    rails_arr = found["rails_arr"]

    table = found["table"]

    mt = found["terms"]

    # Reconstruction bit-exactness over ALL routes.
    failures = 0

    rail_values = (
        RN.bf16_array_to_float32(rails_arr)
        .astype(np.float64)
    )

    for bits_str, terms in table.items():

        total = 0.0

        for rid, sgn in terms:

            total += sgn * rail_values[rid]

        rec = RN.fp32_array_to_bf16_bits(
            np.array([total])
        )[0]

        if rec != np.uint16(int(bits_str)):

            failures += 1

    reconstruction_ok = failures == 0

    # Full-tensor routing coverage.
    known = set(
        int(b) for b in table.keys()
    )

    unrouted = sum(
        1
        for b in np.unique(raw)
        if int(b) not in known
    )

    full_routed = unrouted == 0

    # Fraction mathematical oracle via 05 components.
    topology = ORACLE.RailTopology(
        raw.astype(np.int32),
        table,
        int(shape[1]),
        int(shape[0])
    )

    rng = np.random.default_rng(SEED)

    oracle_ok = True

    x_len = int(shape[1])

    for _v in range(ORACLE_VECTORS):

        x = rng.normal(
            0.0, 1.0, x_len
        ).astype(np.float32)

        x_frac = [
            ORACLE.Fraction.from_float(float(xi))
            for xi in x
        ]

        j_idx = rng.choice(
            int(shape[0]),
            min(
                ORACLE_OUTPUTS,
                int(shape[0])
            ),
            replace=False
        ).tolist()

        dense_t = ORACLE.exact_dense_oracle_rows(
            x_frac,
            raw.reshape(shape),
            j_idx
        )

        rail_fracs = [
            ORACLE.fraction_of_bf16(int(b))
            for b in rails_arr
        ]

        rail_t = ORACLE.exact_rail_oracle_rows(
            x_frac,
            rail_fracs,
            topology,
            j_idx
        )

        if any(
            a != b
            for a, b in zip(dense_t, rail_t)
        ):

            oracle_ok = False

            break

    passed = (
        reconstruction_ok
        and full_routed
        and oracle_ok
    )

    # --------------------------------------------------------
    # Compute census.
    # --------------------------------------------------------

    dense_m, shared_m = census_shared_mults(
        raw,
        table,
        mt,
        int(shape[0]),
        int(shape[1])
    )

    reduction = 1.0 - shared_m / dense_m

    # --------------------------------------------------------
    # Serialize artifact.
    # --------------------------------------------------------

    COMPILED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    artifact_path = (
        COMPILED_DIR
        / f"{short}_lossless.json"
    )

    artifact = build_artifact(
        tensor_name,
        shape,
        rails_arr,
        mt,
        table,
        n_uniq if passed else 0,
        n_uniq,
        {
            "reconstruction": (
                "PASS" if reconstruction_ok
                else f"FAIL({failures})"
            ),
            "full_routed": bool(full_routed),
            "oracle": bool(oracle_ok),
        }
    )

    with open(
        artifact_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(artifact, f)

    # Reload + checksum verification (spec 39).
    cksum_ok, _n = verify_artifact_checksum(
        artifact_path
    )

    print()
    print(
        f"  RESULT {short}: "
        f"rails={found['rails']} terms={mt} "
        f"recon={'OK' if reconstruction_ok else 'FAIL'} "
        f"routed={'OK' if full_routed else 'FAIL'} "
        f"oracle={'EXACT' if oracle_ok else 'FAIL'} "
        f"checksum={'OK' if cksum_ok else 'FAIL'}"
    )

    print(
        f"  COMPUTE dense={dense_m:,} "
        f"shared={shared_m:,} "
        f"reduction={reduction:.2%}"
    )

    print(
        f"  ARTIFACT {artifact_path} "
        f"({artifact_path.stat().st_size:,} bytes)",
        flush=True
    )

    return {
        "tensor": tensor_name,

        "short": short,

        "shape": list(shape),

        "parameters": int(raw.size),

        "unique_values": int(n_uniq),

        "rails": int(found["rails"]),

        "terms": int(mt),

        "exact_unique": int(n_uniq) if passed else 0,

        "pass": bool(passed),

        "dense_multiplications": int(dense_m),

        "shared_multiplications": int(shared_m),

        "multiplication_reduction": float(reduction),

        "artifact_file": str(artifact_path),

        "artifact_bytes": int(
            artifact_path.stat().st_size
        ),

        "checksum_ok": bool(cksum_ok),

        "scan_log": scan_log,
    }


def main():

    only = None

    args = sys.argv[1:]

    if "--only" in args:

        only = args[
            args.index("--only") + 1
        ]

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    all_results = []

    for tensor_name in ATTENTION_TENSORS:

        short = SHORT_NAMES[tensor_name]

        if only and short != only:

            continue

        result = process_tensor(
            tensor_name
        )

        if result is not None:

            all_results.append(result)

        print(flush=True)

    if not all_results:

        print("No results.")

        return

    # ========================================================
    # REQUIRED OUTPUT TABLE (spec 54)
    # ========================================================

    print("=" * 80)
    print("ATTENTION LOSSLESS SUMMARY")
    print("=" * 80)

    header = (
        f"{'Tensor':10s} {'Shape':16s} {'Params':>9s} "
        f"{'Unique':>7s} {'Rails':>6s} {'Terms':>6s} {'Exact':>6s}"
    )

    print(header)
    print("-" * len(header))

    for r in all_results:

        shape_str = (
            f"({r['shape'][0]},{r['shape'][1]})"
        )

        exact_str = (
            f"{r['exact_unique']}/"
            f"{r['unique_values']}"
        )

        status = (
            "PASS" if r["pass"] else "FAIL"
        )

        print(
            f"{r['short']:10s} {shape_str:16s} "
            f"{r['parameters']:>9,} "
            f"{r['unique_values']:>7,} "
            f"{r['rails']:>6d} {r['terms']:>6d} "
            f"{status:>6s} ({exact_str})"
        )

    print()

    total_dense = sum(
        r["dense_multiplications"]
        for r in all_results
    )

    total_shared = sum(
        r["shared_multiplications"]
        for r in all_results
    )

    print(
        f"Dense multiplications : {total_dense:,}"
    )

    print(
        f"Shared multiplications: {total_shared:,}"
    )

    print(
        f"Reduction             : "
        f"{1.0 - total_shared/total_dense:.2%}"
    )

    artifact_bits_total = sum(
        r["artifact_bytes"] * 8
        for r in all_results
    )

    print(
        f"Artifact bits         : "
        f"{artifact_bits_total:,}"
    )

    # Milestone log (spec 61).
    milestone = {
        "milestone": "attention_layer0_lossless",

        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),

        "seed": SEED,

        "results": all_results,

        "totals": {
            "parameters": sum(
                r["parameters"]
                for r in all_results
            ),

            "unique_values": sum(
                r["unique_values"]
                for r in all_results
            ),

            "dense_multiplications": total_dense,

            "shared_multiplications": total_shared,

            "multiplication_reduction": float(
                1.0 - total_shared / total_dense
            ),
        },

        "verdict": (
            "PASS"
            if all(r["pass"] for r in all_results)
            else "INCOMPLETE"
        ),
    }

    out = RESULTS_DIR / "milestone_attention.json"

    with open(
        out,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            milestone,
            f,
            indent=2
        )

    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
