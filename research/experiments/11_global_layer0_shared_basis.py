import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 11-GLOBAL
# LAYER-0 GLOBAL SHARED RAIL BASIS EXPERIMENT (spec 17-19)
#
# Question:
#   Can ALL seven Layer-0 tensors be represented 100% exact
#   with ONE shared rail dictionary instead of per-tensor
#   dictionaries?
#
# Method:
#   - union of unique BF16 values across all 7 tensors
#     (weighted by combined frequency)
#   - ONE learned global basis (uniform-rank init +
#     coordinate descent + missing-value/safe-slot repairs)
#   - success criterion: union exact == 100%
#     (=> every tensor is exact, since tensor sets ⊆ union)
#
# Scale engineering:
#   - vectorized meet-in-the-middle 4-term exhaustive search
#     replaces the per-target linear pair scan, then is
#     monkeypatched into the compiler module so that all
#     repair stages automatically use the fast path.
# ============================================================


MODEL_FILE = Path("model_data/model.safetensors")

LAYER0_TENSORS = [
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
]

RAIL_GRID_DEFAULT = [256, 384, 512]

TERMS = 4

SEED = 42

COMPILED_DIR = Path("compiled") / "layer0"

RESULTS_DIR = Path("results")


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
# FAST EXHAUSTIVE COMPILE (meet-in-the-middle, max_terms<=4)
# Drop-in replacement for RN.compile_exact_routes_exhaustive
# ============================================================

def make_fast_compile():

    state = {}

    def fast_compile(
        unique_bits,
        rails,
        max_terms
    ):

        t0 = time.perf_counter()

        target_bits = [
            int(x) for x in unique_bits
        ]

        rail_values = (
            RN.bf16_array_to_float32(rails)
            .astype(np.float64)
        )

        # ---- 1-term & pair table (same as baseline) ----

        table = {}

        add = RN._add_candidate

        for i in range(len(rails)):

            vi = float(rail_values[i])

            add(table, vi, ((i, 1),))

            add(table, -vi, ((i, -1),))

        for i in range(len(rails)):

            vi = float(rail_values[i])

            for j in range(i + 1, len(rails)):

                vj = float(rail_values[j])

                add(table, vi + vj, ((i, 1), (j, 1)))

                add(table, vi - vj, ((i, 1), (j, -1)))

                add(table, -vi + vj, ((i, -1), (j, 1)))

                add(table, -vi - vj, ((i, -1), (j, -1)))

        routes = {}

        def bf16_val(bits):

            return float(
                RN.bf16_bits_to_float32(int(bits))
            )

        def try_route(bits, cand):

            if cand is None:

                return False

            if not RN._route_has_unique_rails(cand):

                return False

            total = 0.0

            rv = rail_values

            for rid, sgn in cand:

                total += sgn * rv[rid]

            return int(bits) == int(
                RN.float32_to_bf16_bits(total)
            )

        # 1-term
        for rid in range(len(rails)):

            v = float(rail_values[rid])

            routes[
                int(RN.float32_to_bf16_bits(v))
            ] = ((rid, 1),)

            routes[
                int(RN.float32_to_bf16_bits(-v))
            ] = ((rid, -1),)

        # Sorted pair arrays for vectorized 4-term MITM.
        pair_items = sorted(table.items())

        A = np.array(
            [p[0] for p in pair_items],
            dtype=np.float64
        )

        A_sorted = np.sort(A)

        sorted_routes = []

        order = np.argsort(A)

        for idx in order:

            sorted_routes.append(
                pair_items[idx][1]
            )

        # 2-term
        for bits in target_bits:

            b = int(bits)

            if b in routes:

                continue

            cand = table.get(bf16_val(bits))

            if try_route(b, cand):

                routes[b] = cand

        # 3-term
        for bits in target_bits:

            b = int(bits)

            if b in routes:

                continue

            t = bf16_val(bits)

            found = None

            for rid in range(len(rails)):

                rvv = float(rail_values[rid])

                for sign in (1, -1):

                    rem = t - sign * rvv

                    pr = table.get(rem)

                    if pr is None:

                        continue

                    cand = pr + ((rid, sign),)

                    if try_route(b, cand):

                        found = cand

                        break

                if found:

                    break

            if found:

                routes[b] = found

        # 4-term: vectorized meet-in-the-middle
        remaining = [
            b for b in target_bits
            if int(b) not in routes
        ]

        for bits in remaining:

            b = int(bits)

            t = bf16_val(bits)

            comps = t - A_sorted

            pos = np.searchsorted(
                A_sorted,
                comps
            )

            pos_clipped = np.minimum(
                pos,
                len(A_sorted) - 1
            )

            hit = (
                A_sorted[pos_clipped]
                == comps
            )

            hit_idx = np.flatnonzero(hit)

            found = None

            for hi in hit_idx:

                p = float(A_sorted[hi])

                ra = sorted_routes[hi]

                rb = table.get(t - p)

                if rb is None:

                    continue

                cand = ra + rb

                if not cand or len(cand) > max_terms:

                    continue

                if try_route(b, cand):

                    found = cand

                    break

            if found:

                routes[b] = found

        elapsed = time.perf_counter() - t0

        print(
            f"    [fast-compile R={len(rails)}] "
            f"covered {len(routes)} "
            f"({elapsed:.1f}s)",
            flush=True
        )

        state["last_seconds"] = elapsed

        return routes

    return fast_compile


FAST_COMPILE = make_fast_compile()

# Monkeypatch compiler + oracle modules so every internal
# call site (repairs, final refresh) uses the fast path.
RN.compile_exact_routes_exhaustive = FAST_COMPILE

RN.MAX_ITERS = 4

RN.REPAIR_COMPILE_BUDGET = 140

RN.SAFE_SCAN_LIMIT = 140


# ============================================================
# UNION DATASET
# ============================================================

def build_union():

    per_tensor = {}

    chunks = []

    for name in LAYER0_TENSORS:

        RN.TARGET_TENSOR = name

        raw, shape = RN.read_target_tensor()

        per_tensor[name] = {
            "raw": raw,
            "shape": shape,
        }

        chunks.append(raw)

    merged = np.concatenate(chunks)

    del chunks

    uniq, counts = np.unique(
        merged,
        return_counts=True
    )

    vals = (
        RN.bf16_array_to_float32(uniq)
        .astype(np.float64)
    )

    print(
        f"UNION: {merged.size:,} params -> "
        f"{uniq.size:,} unique values"
    )

    return per_tensor, uniq, counts.astype(np.float64), vals


# ============================================================
# MAIN
# ============================================================

def main():

    rail_grid = RAIL_GRID_DEFAULT

    args = sys.argv[1:]

    if "--rails" in args:

        rail_grid = [
            int(x)
            for x in args[
                args.index("--rails") + 1
            ].split(",")
        ]

    print("=" * 80)
    print("RAILNET LAYER-0 GLOBAL SHARED RAIL BASIS")
    print(f"grid: {rail_grid} x terms={TERMS}")
    print("=" * 80, flush=True)

    per_tensor, uniq, counts, vals = (
        build_union()
    )

    n_uniq = uniq.size

    winner = None

    log = []

    for rc in rail_grid:

        print()
        print(f"--- GLOBAL RAILS = {rc} ---", flush=True)

        start = time.perf_counter()

        learned = RN.learn_basis(
            vals,
            uniq,
            counts,
            rc,
            TERMS
        )

        elapsed = time.perf_counter() - start

        table = FAST_COMPILE(
            uniq,
            learned["rails"],
            TERMS
        )

        cov = sum(
            1
            for b in uniq
            if int(b) in table
        )

        lossless = cov == n_uniq

        log.append(
            {
                "rails": rc,
                "exact": int(cov),
                "lossless": bool(lossless),
                "seconds": round(elapsed, 1),
            }
        )

        print(
            f"GLOBAL R={rc}: exact {cov}/{n_uniq} "
            f"({cov/n_uniq:.3%}) "
            f"{'LOSSLESS' if lossless else ''} "
            f"[{elapsed:.0f}s]",
            flush=True
        )

        if lossless:

            winner = {
                "rails": rc,
                "rails_arr": learned["rails"],
                "table": table,
            }

            break

    if winner is None:

        print()
        print("NO LOSSLESS GLOBAL BASIS IN GRID")

        RESULTS_DIR.mkdir(exist_ok=True)

        json.dump(
            {
                "experiment": "layer0_global_shared_rail",

                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),

                "grid": log,

                "verdict": "NO_LOSSLESS_IN_GRID",
            },
            open(
                RESULTS_DIR / "milestone_global_basis.json",
                "w"
            ),
            indent=2
        )

        return

    # --------------------------------------------------------
    # Per-tensor validation with the GLOBAL basis.
    # --------------------------------------------------------

    rails_arr = winner["rails_arr"]

    table = winner["table"]

    rail_fracs = None

    per_tensor_results = []

    total_dense = 0

    total_shared = 0

    rng = np.random.default_rng(SEED)

    for name in LAYER0_TENSORS:

        entry = per_tensor[name]

        raw = entry["raw"]

        shape = entry["shape"]

        known = set(
            int(b) for b in table.keys()
        )

        unrouted = sum(
            1
            for b in np.unique(raw)
            if int(b) not in known
        )

        full_routed = unrouted == 0

        # Fraction oracle sample.
        topology = ORACLE.RailTopology(
            raw.astype(np.int32),
            table,
            int(shape[1]),
            int(shape[0])
        )

        oracle_ok = True

        x_len = int(shape[1])

        for _v in range(2):

            x = rng.normal(
                0.0, 1.0, x_len
            ).astype(np.float32)

            x_frac = [
                ORACLE.Fraction.from_float(
                    float(xi)
                )
                for xi in x
            ]

            j_idx = rng.choice(
                int(shape[0]),
                min(24, int(shape[0])),
                replace=False
            ).tolist()

            dense_t = (
                ORACLE.exact_dense_oracle_rows(
                    x_frac,
                    raw.reshape(shape),
                    j_idx
                )
            )

            if rail_fracs is None:

                rail_fracs = [
                    ORACLE.fraction_of_bf16(
                        int(b)
                    )
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

        # Shared multiplication census (bit-indexed).
        term_max = max(
            len(t) for t in table.values()
        )

        term_rail = np.zeros(
            (65_536, term_max),
            dtype=np.int32
        )

        term_active = np.zeros(
            (65_536, term_max),
            dtype=bool
        )

        for bits_str, terms in table.items():

            g = int(bits_str)

            for t_i, (rid, _s) in enumerate(terms):

                term_rail[g, t_i] = rid

                term_active[g, t_i] = True

        pairs = set()

        n_elems = raw.size

        chunk = 1_000_000

        input_size = int(shape[1])

        for s in range(0, n_elems, chunk):

            e = min(s + chunk, n_elems)

            g = raw[s:e].astype(np.int32)

            idx = np.arange(
                s, e, dtype=np.int64
            )

            jj = idx // input_size

            for t_i in range(term_max):

                act = term_active[g, t_i]

                if not np.any(act):

                    continue

                rr = term_rail[g[act], t_i]

                pairs.update(
                    (
                        jj[act]
                        * 70_000
                        + rr
                    ).tolist()
                )

        dense_m = int(n_elems)

        shared_m = len(pairs)

        total_dense += dense_m

        total_shared += shared_m

        ok = full_routed and oracle_ok

        per_tensor_results.append(
            {
                "tensor": name,

                "shape": list(shape),

                "parameters": dense_m,

                "full_tensor_exact": bool(full_routed),

                "math_oracle_exact": bool(oracle_ok),

                "shared_multiplications": shared_m,

                "dense_multiplications": dense_m,

                "reduction": float(
                    1.0 - shared_m / dense_m
                ),

                "pass": bool(ok),
            }
        )

        print(
            f"  {name.split('.')[-2]:9s} routed="
            f"{'OK' if full_routed else 'FAIL'} "
            f"oracle={'EXACT' if oracle_ok else 'FAIL'} "
            f"shared={shared_m:,}",
            flush=True
        )

    # --------------------------------------------------------
    # Serialize GLOBAL artifact.
    # --------------------------------------------------------

    content = {
        "magic": "RNET_GLOBAL",

        "version": 1,

        "dtype": "BF16",

        "scope": "model.layers.0 (7 tensors)",

        "tensors": [
            n for n in LAYER0_TENSORS
        ],

        "rail_count": int(winner["rails"]),

        "max_terms": TERMS,

        "rails": [
            int(b) for b in rails_arr
        ],

        "routes": {
            str(b): [
                [int(rid), int(sgn)]
                for rid, sgn in r
            ]
            for b, r in sorted(
                table.items()
            )
        },

        "per_tensor_validation": per_tensor_results,
    }

    canonical = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")

    content["checksum_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()

    COMPILED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = (
        COMPILED_DIR
        / "_GLOBAL_layer0.json"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(content, f)

    # ========================================================
    # REQUIRED OUTPUT (spec 55)
    # ========================================================

    total_params = sum(
        r["parameters"]
        for r in per_tensor_results
    )

    all_pass = all(
        r["pass"]
        for r in per_tensor_results
    )

    print()
    print("=" * 80)
    print("Layer-0 GLOBAL RAIL FABRIC")
    print("=" * 80)

    print(f"Tensors          : {len(LAYER0_TENSORS)}")

    print(f"Total parameters : {total_params:,}")

    print(
        f"Global unique    : {n_uniq:,}"
    )

    print(
        f"Rails            : {winner['rails']} "
        f"(terms={TERMS})"
    )

    print(
        f"Exact            : "
        f"{'100% ALL TENSORS' if all_pass else 'INCOMPLETE'}"
    )

    print(
        f"Dense multiply   : {total_dense:,}"
    )

    print(
        f"Shared multiply  : {total_shared:,}"
    )

    print(
        f"Reduction        : "
        f"{1.0 - total_shared/total_dense:.2%}"
    )

    # --------------------------------------------------------
    # HONEST full representation accounting (spec 20/42/60).
    #
    # The artifact FILE size is the dictionary only.
    # The dominant component is the per-element route map:
    # every tensor element needs its route id (or raw BF16
    # key under the current bit-indexed runtime convention).
    # --------------------------------------------------------

    import math

    n_routes_global = len(table)

    id_bits = math.ceil(
        math.log2(max(n_routes_global, 2))
    )

    active_terms_global = sum(
        len(t) for t in table.values()
    )

    comp_rails = winner["rails"] * 16

    comp_desc = (
        active_terms_global
        * (id_bits + 1)
    )

    comp_map_13 = total_params * id_bits

    comp_map_16 = total_params * 16

    comp_meta = 4096

    honest_13 = (
        comp_rails
        + comp_desc
        + comp_map_13
        + comp_meta
    )

    honest_16 = (
        comp_rails
        + comp_desc
        + comp_map_16
        + comp_meta
    )

    orig_bits_total = (
        total_params * 16
    )

    print()
    print(
        f"FULL REPRESENTATION (honest):"
    )

    print(
        f"  A rails             : {comp_rails:,} b"
    )

    print(
        f"  B route descriptions: {comp_desc:,} b "
        f"({n_routes_global:,} routes, "
        f"{active_terms_global:,} terms)"
    )

    print(
        f"  C element map 13-bit: {comp_map_13:,} b  "
        f"<-- DOMINANT"
    )

    print(
        f"  C element map 16-bit: {comp_map_16:,} b "
        f"(raw-key convention)"
    )

    print(
        f"  D metadata          : {comp_meta:,} b"
    )

    print(
        f"  HONEST total (13b)  : {honest_13:,} b "
        f"-> storage compression "
        f"{orig_bits_total/honest_13:.3f}x"
    )

    print(
        f"  HONEST total (16b)  : {honest_16:,} b "
        f"-> {orig_bits_total/honest_16:.4f}x "
        f"(~no bit-storage gain; mapping IS "
        f"the fabric configuration)"
    )

    print(
        f"  [dictionary-only]   : "
        f"{out_path.stat().st_size * 8:,} b "
        f"(NOT a valid compression metric)"
    )

    print()

    print(
        f"Runtime weight array: ABSENT "
        f"(resident rails+topology ~= "
        f"{(comp_rails+comp_desc)/8/1024:.1f} KiB)"
    )

    milestone = {
        "milestone": "layer0_global_shared_rail",

        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),

        "seed": SEED,

        "grid_log": log,

        "winner_rails": winner["rails"],

        "terms": TERMS,

        "union_unique_values": int(n_uniq),

        "per_tensor": per_tensor_results,

        "totals": {
            "parameters": total_params,

            "dense_multiplications": total_dense,

            "shared_multiplications": total_shared,

            "reduction": float(
                1.0 - total_shared / total_dense
            ),
        },

        "artifact": str(out_path),

        "full_representation_honest": {
            "note": (
                "Artifact file size is dictionary-only. "
                "Dominant component is per-element route map."
            ),

            "components_bits": {
                "rails": comp_rails,

                "route_descriptions": comp_desc,

                "element_map_13bit_ids": comp_map_13,

                "element_map_16bit_raw_keys": comp_map_16,

                "metadata_estimate": comp_meta,
            },

            "original_dense_bits": orig_bits_total,

            "honest_total_bits_13bit_ids": honest_13,

            "storage_compression_13bit_ids": float(
                orig_bits_total / honest_13
            ),

            "honest_total_bits_16bit_raw_keys": honest_16,

            "storage_compression_16bit_raw_keys": float(
                orig_bits_total / honest_16
            ),
        },

        "verdict": (
            "PASS" if all_pass else "PARTIAL"
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)

    with open(
        RESULTS_DIR / "milestone_global_basis.json",
        "w"
    ) as f:

        json.dump(milestone, f, indent=2)

    print()
    print(
        "Saved: results/milestone_global_basis.json"
    )


if __name__ == "__main__":
    main()
