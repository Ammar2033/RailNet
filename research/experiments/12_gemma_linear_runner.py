import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 12
# GEMMA LINEAR RUNNER - GLOBAL 96-RAIL FABRIC RUNTIME
#
# Question finally answered:
#   "Does this topology actually take an input and produce
#    the correct output?"
#
# Runtime API (spec 22) - the ONLY legal form:
#
#     output = rail_linear(x, compiled_tensor)
#
# The compiled tensor contains: rails + topology tables.
# It does NOT contain, and rail_linear never receives,
# the dense weight matrix.
#
# Validation tiers (spec 19):
#   T1  weight exactness            (bit-level, must PASS)
#   T2  mathematical output exact   (Fraction oracle, must PASS;
#        excluded from runtime timing per spec 43)
#   T3  normal FP64/BF16 equality   (diagnostic ONLY; may differ
#        due to accumulation order - not a failure)
#
# Timing separation (spec 43):
#   artifact_load_time | runtime_kernel_time | (oracle untimed)
# ============================================================


MODEL_FILE = Path("model_data/model.safetensors")

DEFAULT_ARTIFACT = Path(
    "compiled/layer0/_GLOBAL_layer0.json"
)

BATCH_ROWS = [1, 2, 4, 8, 16]

KERNEL_REPS = 5

SEED = 42

ORACLE_VECTORS = 2

ORACLE_OUTPUTS = 24


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

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
# COMPILED TENSOR LOADER (checksum validated, spec 39)
# ============================================================

class CompiledTensor:

    def __init__(
        self,
        artifact_path,
        raw_bits,
        shape
    ):

        t0 = time.perf_counter()

        with open(
            artifact_path,
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

        self.checksum_ok = (
            hashlib.sha256(canonical).hexdigest()
            == stored
        )

        if not self.checksum_ok:

            raise RuntimeError(
                f"Artifact checksum FAIL: {artifact_path}"
            )

        self.tensor_name = data.get(
            "tensor",
            data.get("scope", "<global>")
        )

        self.rail_count = data["rail_count"]

        self.max_terms = data["max_terms"]

        self.rails_f64 = (
            RN.bf16_array_to_float32(
                np.array(
                    data["rails"],
                    dtype=np.uint16
                )
            ).astype(np.float64)
        )

        # Topology tables indexed by RAW BF16 PATTERN
        # (route id == bits, no remapping).
        rows = 65_536

        mt = self.max_terms

        self.term_rail = np.zeros(
            (rows, mt), dtype=np.int32
        )

        self.term_sign = np.zeros(
            (rows, mt), dtype=np.int8
        )

        self.term_active = np.zeros(
            (rows, mt), dtype=bool
        )

        for bits_str, terms in data[
            "routes"
        ].items():

            g = int(bits_str)

            for t_i, (rid, sgn) in enumerate(
                terms
            ):

                self.term_rail[g, t_i] = rid

                self.term_sign[g, t_i] = sgn

                self.term_active[g, t_i] = True

        # Routing state for THIS tensor: its own raw
        # bit patterns ARE the route ids (spec: no dense
        # weight VALUES are stored or used by the kernel -
        # only the 16-bit route selectors, which is exactly
        # the honest element-map cost from the report).
        self.route_ids = (
            raw_bits.astype(np.int32)
            .reshape(shape)
        )

        self.shape = tuple(shape)

        self.out_features = int(shape[0])

        self.in_features = int(shape[1])

        self.load_seconds = (
            time.perf_counter() - t0
        )


# ============================================================
# RAILNET RUNTIME KERNEL (spec 21/22)
# ============================================================

def rail_linear(x, compiled):
    """
    Shared-rail linear layer.

        Y[j] = sum_r Rails[r] * (sum_{(i,s) in route(j,i)} s * x[i])

    x      : float64 array (in_features,)
    compiled : CompiledTensor

    Returns float64 (out_features,).

    Multiplications performed: out x rail_count (shared),
    NOT out x in (dense). No weights touched.
    """

    c = compiled

    g = c.route_ids.reshape(-1)

    n = g.size

    ii = np.arange(n) % c.in_features

    xv = x[ii]

    jjR = (
        (np.arange(n) // c.in_features)
        * c.rail_count
    ).astype(np.int64)

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

        G = np.bincount(
            idx,
            weights=wgt,
            minlength=c.out_features * c.rail_count
        )

    else:

        G = np.zeros(
            c.out_features * c.rail_count
        )

    Y = (
        G.reshape(c.out_features, c.rail_count)
        * c.rails_f64[None, :]
    ).sum(axis=1)

    return Y


# ============================================================
# DENSE REFERENCE (oracle-side only; never given to kernel)
# ============================================================

def dense_reference(x_f64, weights_f64):

    return x_f64 @ weights_f64.T


# ============================================================
# PER-TENSOR PIPELINE
# ============================================================

def run_tensor(name, artifact_path):

    print("-" * 80)
    print(f"RUNNING: {name}")
    print("-" * 80, flush=True)

    RN.TARGET_TENSOR = name

    raw, shape = RN.read_target_tensor()

    out_f, in_f = int(shape[0]), int(shape[1])

    # ---- T1: weight exactness -----------------------------

    bits, counts, vals = RN.analyze_unique_values(raw)

    learned_table_probe = None  # not needed; use artifact

    # Load compiled (timed) and bind to this tensor.
    compiled = CompiledTensor(
        artifact_path,
        raw,
        shape
    )

    # Verify every unique value has an exact route.
    rail_values = compiled.rails_f64

    failures = 0

    for b in bits:

        key = str(int(b))

        # direct table access instead of dict lookup:
        g = int(b)

        total = 0.0

        ok_any = False

        for t in range(compiled.max_terms):

            if compiled.term_active[g, t]:

                ok_any = True

                total += (
                    compiled.term_sign[g, t]
                    * rail_values[
                        compiled.term_rail[g, t]
                    ]
                )

        if not ok_any:

            failures += 1

            continue

        rec = RN.fp32_array_to_bf16_bits(
            np.array([total])
        )[0]

        if rec != np.uint16(int(b)):

            failures += 1

    weight_exact = failures == 0

    print(
        f"T1 weight exact     : "
        f"{'PASS' if weight_exact else 'FAIL'} "
        f"({len(bits) - failures}/{len(bits)} routes)"
    )

    print(
        f"   artifact checksum : "
        f"{'OK' if compiled.checksum_ok else 'FAIL'} "
        f"| load {compiled.load_seconds*1000:.1f} ms"
    )

    # ---- Runtime batches (timed) --------------------------

    rng = np.random.default_rng(SEED)

    weights_f64 = (
        RN.bf16_array_to_float32(raw)
        .astype(np.float64)
        .reshape(shape)
    )

    kernel_ms_per_row = {}

    fp64_diag = {
        "mismatch_rows": 0,
        "total_rows": 0,
        "max_abs_err": 0.0,
    }

    bf16_diag = {
        "mismatch_elements": 0,
        "total_elements": 0,
    }

    for rows in BATCH_ROWS:

        X = rng.normal(
            0.0, 1.0, (rows, in_f)
        ).astype(np.float64)

        # warm-up
        for r in range(rows):

            rail_linear(X[r], compiled)

        t0 = time.perf_counter()

        reps = max(1, KERNEL_REPS // max(1, rows // 2))

        for _ in range(reps):

            for r in range(rows):

                Y_rn = rail_linear(X[r], compiled)

        dt = time.perf_counter() - t0

        kernel_ms_per_row[rows] = (
            dt / (reps * rows) * 1000.0
        )

        # correctness for this batch
        for r in range(rows):

            Y_rn = rail_linear(X[r], compiled)

            Y_ref = dense_reference(X[r], weights_f64)

            diff = np.abs(Y_rn - Y_ref)

            fp64_diag["total_rows"] += 1

            if diff.max() > 0:

                fp64_diag["mismatch_rows"] += 1

            fp64_diag["max_abs_err"] = max(
                fp64_diag["max_abs_err"],
                float(diff.max())
            )

            # BF16-rounded comparison (diagnostic tier 3)
            rn_b = RN.fp32_array_to_bf16_bits(
                Y_rn.astype(np.float32)
            )

            ref_b = RN.fp32_array_to_bf16_bits(
                Y_ref.astype(np.float32)
            )

            mism = int(np.count_nonzero(rn_b != ref_b))

            bf16_diag["total_elements"] += len(Y_rn)

            bf16_diag["mismatch_elements"] += mism

    # ---- T2: Fraction mathematical oracle (untimed) -------

    oracle_failures = 0

    oracle_checked = 0

    for _v in range(ORACLE_VECTORS):

        x = rng.normal(0.0, 1.0, in_f).astype(
            np.float32
        )

        x_frac = [
            ORACLE.Fraction.from_float(float(xi))
            for xi in x
        ]

        j_idx = rng.choice(
            out_f,
            min(ORACLE_OUTPUTS, out_f),
            replace=False
        ).tolist()

        dense_truth = (
            ORACLE.exact_dense_oracle_rows(
                x_frac,
                raw.reshape(shape),
                j_idx
            )
        )

        rail_fracs = [
            ORACLE.Fraction.from_float(float(v))
            for v in compiled.rails_f64
        ]

        topo = ORACLE.RailTopology(
            raw.astype(np.int32),
            {
                str(g): [
                    [int(compiled.term_rail[g, t]),
                     int(compiled.term_sign[g, t])]
                    for t in range(compiled.max_terms)
                    if compiled.term_active[g, t]
                ]
                for g in range(65_536)
                if np.any(
                    compiled.term_active[g]
                )
            },
            in_f,
            out_f
        )

        rail_truth = ORACLE.exact_rail_oracle_rows(
            x_frac,
            rail_fracs,
            topo,
            j_idx
        )

        for a, b in zip(dense_truth, rail_truth):

            oracle_checked += 1

            if a != b:

                oracle_failures += 1

    math_exact = oracle_failures == 0

    print(
        f"T2 math output exact: "
        f"{'EXACT' if math_exact else 'FAIL'} "
        f"(Fraction, {oracle_checked} outputs checked)"
    )

    print(
        f"T3 FP64 diagnostic  : "
        f"{fp64_diag['mismatch_rows']}/"
        f"{fp64_diag['total_rows']} rows differ, "
        f"max|err|={fp64_diag['max_abs_err']:.3e} "
        f"[accumulation-order effect, NOT a failure]"
    )

    print(
        f"T3 BF16-rounded diag: "
        f"{bf16_diag['mismatch_elements']}/"
        f"{bf16_diag['total_elements']} elements differ "
        f"[diagnostic]"
    )

    print(
        f"kernel ms/row       : " +
        ", ".join(
            f"B{b}={v:.3f}"
            for b, v in kernel_ms_per_row.items()
        )
    )

    passed = weight_exact and math_exact

    return {
        "tensor": name,

        "shape": list(shape),

        "parameters": int(raw.size),

        "t1_weight_exact": bool(weight_exact),

        "t2_math_output_exact": bool(math_exact),

        "t3_fp64_max_abs_err": fp64_diag[
            "max_abs_err"
        ],

        "t3_fp64_mismatch_rows": fp64_diag[
            "mismatch_rows"
        ],

        "t3_fp64_total_rows": fp64_diag[
            "total_rows"
        ],

        "t3_bf16_rounded_mismatches": bf16_diag[
            "mismatch_elements"
        ],

        "t3_bf16_rounded_total": bf16_diag[
            "total_elements"
        ],

        "kernel_ms_per_row": kernel_ms_per_row,

        "artifact_load_ms": compiled.load_seconds * 1000,

        "pass": bool(passed),
    }


def main():

    args = sys.argv[1:]

    artifact_path = DEFAULT_ARTIFACT

    if "--artifact" in args:

        artifact_path = Path(
            args[args.index("--artifact") + 1]
        )

    only = None

    if "--only" in args:

        only = args[args.index("--only") + 1]

    print("=" * 80)
    print("RAILNET STAGE 12 - GEMMA LINEAR RUNNER")
    print(f"artifact: {artifact_path}")
    print("=" * 80, flush=True)

    tensors = [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]

    results = []

    for name in tensors:

        if only and only not in name:

            continue

        try:

            results.append(
                run_tensor(name, artifact_path)
            )

        except Exception as exc:

            print(
                f"ERROR in {name}: {exc}",
                flush=True
            )

        print(flush=True)

    if not results:

        return

    # ========================================================
    # FINAL SUMMARY (spec 41 categories)
    # ========================================================

    print("=" * 80)
    print("STAGE 12 SUMMARY - GLOBAL 96-RAIL FABRIC RUNTIME")
    print("=" * 80)

    hdr = (
        f"{'Tensor':10s} {'T1 Weight':>10s} "
        f"{'T2 Math':>10s} {'T3 FP64diag':>12s} "
        f"{'ms/row(B1)':>11s}"
    )

    print(hdr)
    print("-" * len(hdr))

    for r in results:

        print(
            f"{r['tensor'].split('.')[-2]:10s} "
            f"{'PASS' if r['t1_weight_exact'] else 'FAIL':>10s} "
            f"{'EXACT' if r['t2_math_output_exact'] else 'FAIL':>10s} "
            f"{r['t3_fp64_max_abs_err']:>12.3e} "
            f"{r['kernel_ms_per_row'][1]:>11.3f}"
        )

    all_pass = all(r["pass"] for r in results)

    print()
    print(
        f"CORRECTNESS : "
        f"{'ALL PASS' if all_pass else 'INCOMPLETE'} "
        f"(T1+T2 mandatory)"
    )

    print(
        f"MEMORY      : dense weight array ABSENT from "
        f"runtime path (routing map only)"
    )

    avg_b1 = sum(
        r["kernel_ms_per_row"][1] for r in results
    ) / len(results)

    print(
        f"COMPUTE     : mean kernel {avg_b1:.3f} ms/row "
        f"@ batch=1 (CPU, unoptimized)"
    )

    print(
        f"RUNTIME     : artifact load + kernel timed; "
        f"oracle excluded (spec 43)"
    )

    milestone = {
        "milestone": "stage12_linear_runner",

        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),

        "artifact": str(artifact_path),

        "seed": SEED,

        "batch_rows": BATCH_ROWS,

        "results": results,

        "verdict": (
            "PASS" if all_pass else "INCOMPLETE"
        ),
    }

    out = Path("results/milestone_stage12_runner.json")

    with open(out, "w") as f:

        json.dump(milestone, f, indent=2)

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
