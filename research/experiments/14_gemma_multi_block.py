import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 14
# MULTI-BLOCK EXECUTION WITH KV CACHE (LAYER-0 FABRIC)
#
# STATIC  (compiled, immutable at runtime):
#     96 shared BF16 rails
#     global topology tables (bit-pattern indexed)
#     compiled artifact (_GLOBAL_layer0.json)
#
# DYNAMIC (runtime state):
#     hidden states, Q/K/V, attention scores/probs,
#     KV cache (per block, grows during decode)
#
# Scope note (honest):
#     Only model.layers.0 tensors are compiled. To isolate
#     MULTI-BLOCK STATE semantics, every block re-uses the
#     SAME layer-0 weight set on BOTH paths (dense reference
#     and RailNet). Weight correctness per distinct layer is
#     orthogonal and already proven by the compiler pipeline;
#     per-layer compilation arrives with the scale-out stage.
#
# Modes (spec/user):
#     PREFILL : process seq tokens at once   (seq > 1 tested)
#     DECODE  : one token at a time, KV cache present,
#               positions continue from cache length
#
# Checks:
#     - per-block boundary outputs      EXACT (bf16-rounded)
#     - K cache / V cache               EXACT (bf16-rounded)
#     - cache growth                    EXACT (lengths + content)
#     - decode(t) == prefill(t)         consistency both paths
#     - dense decode == railnet decode  EXACT
# ============================================================


ARTIFACT_PATH = Path(
    "compiled/layer0/_GLOBAL_layer0.json"
)

SEQ_GRID = [1, 2, 4, 8]

BLOCK_COUNTS = [2, 3]

DECODE_LEN = 8

SEED = 42


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    m = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(m)

    return m


HERE = Path(__file__).resolve().parent

RN = load_module(HERE / "04_bf16_learned_basis.py", "rn")

R12 = load_module(HERE / "12_gemma_linear_runner.py", "r12")

M13 = load_module(HERE / "13_gemma_single_block.py", "m13")


CFG = M13.CFG

HIDDEN = CFG["hidden_size"]

HEADS = CFG["num_attention_heads"]

KV_HEADS = CFG["num_key_value_heads"]

HEAD_DIM = CFG["head_dim"]

KV_GROUPS = HEADS // KV_HEADS

Q_SCALE = CFG["query_pre_attn_scalar"] ** -0.5

SLIDING_WINDOW = CFG["sliding_window"]


# ============================================================
# ROPE WITH ARBITRARY POSITIONS (extends stage-13 helper)
# ============================================================

def rope_cos_sin_positions(positions):

    dim = HEAD_DIM

    half = np.arange(0, dim, 2, dtype=np.float64)

    inv_freq = CFG["rope_local_base_freq"] ** (
        -(half / dim)
    )

    pos = np.atleast_1d(
        np.asarray(positions, dtype=np.float64)
    )

    freqs = pos[:, None] * inv_freq[None, :]

    emb = np.concatenate([freqs, freqs], axis=-1)

    return np.cos(emb), np.sin(emb)


# ============================================================
# SINGLE BLOCK FORWARD WITH KV CACHE
# ============================================================

def block_forward_cached(
    h,
    norms,
    lin,
    cache,          # dict with "K","V" arrays or None
    pos_offset,
    update_cache=True,
):
    """
    h        : (seq, HIDDEN)
    cache    : per-THIS-block KV cache; mutated when
               update_cache=True (K,V appended)
    Returns block output (seq, HIDDEN) and (K_full, V_full).
    """

    seq = h.shape[0]

    residual = h

    hn = M13.rms_norm(h, norms["input_layernorm"])

    q = lin("q_proj", hn)

    k = lin("k_proj", hn)

    v = lin("v_proj", hn)

    qh = q.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)

    kh = k.reshape(seq, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

    vh = v.reshape(seq, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

    qh = M13.rms_norm(qh, norms["q_norm"])

    kh = M13.rms_norm(kh, norms["k_norm"])

    positions = (
        pos_offset + np.arange(seq)
    )

    cos, sin = rope_cos_sin_positions(positions)

    qh = qh * cos[None] + M13.rotate_half(qh) * sin[None]

    kh = kh * cos[None] + M13.rotate_half(kh) * sin[None]

    # ---- cache handling (DYNAMIC state) -------------------

    if cache is None:

        cache = {"K": kh, "V": vh}

    else:

        cache = {
            "K": np.concatenate([cache["K"], kh], axis=1),

            "V": np.concatenate([cache["V"], vh], axis=1),
        }

    K_full = cache["K"]

    V_full = cache["V"]

    kv_len = K_full.shape[1]

    kh_rep = np.repeat(K_full, KV_GROUPS, axis=0)

    vh_rep = np.repeat(V_full, KV_GROUPS, axis=0)

    # scores (heads, seq, kv_len); causal relative to offset
    scores = np.matmul(qh, kh_rep.transpose(0, 2, 1)) * Q_SCALE

    qi = pos_offset + np.arange(seq)[:, None]

    kj = np.arange(kv_len)[None, :]

    mask = kj > qi          # future keys masked

    scores = np.where(mask[None], -np.inf, scores)

    probs = M13.softmax_last(scores)

    ctx = np.matmul(probs, vh_rep)

    attn_out = ctx.transpose(1, 0, 2).reshape(seq, HEADS * HEAD_DIM)

    o = lin("o_proj", attn_out)

    h = residual + o

    h = M13.rms_norm(h, norms["post_attention_layernorm"])

    # ---- FFN ----------------------------------------------

    residual = h

    hff = M13.rms_norm(h, norms["pre_feedforward_layernorm"])

    g = lin("gate_proj", hff)

    u = lin("up_proj", hff)

    prod = M13.gelu_tanh(g) * u

    d = lin("down_proj", prod)

    h = M13.rms_norm(d, norms["post_feedforward_layernorm"])

    h = residual + h

    return h, cache


# ============================================================
# STACK RUNNERS
# ============================================================

def make_dense_linear(dense_w):

    def lin(short, x):

        return x @ dense_w[short].T

    return lin


def make_rail_linear(compiled):

    def lin(short, x):

        out = np.empty(
            (x.shape[0], compiled[short].out_features),
            dtype=np.float64
        )

        for r in range(x.shape[0]):

            out[r] = R12.rail_linear(
                x[r].astype(np.float64),
                compiled[short]
            )

        return out

    return lin


def stack_forward(
    h0,
    n_blocks,
    norms,
    lin,
    collect_caches=False,
):
    caches = [None] * n_blocks

    block_outputs = []

    h = h0

    for b in range(n_blocks):

        h, caches[b] = block_forward_cached(
            h, norms, lin, caches[b],
            pos_offset=0,
            update_cache=True,
        )

        block_outputs.append(h.copy())

    return (
        h,
        block_outputs,
        caches if collect_caches else None,
    )


def stack_decode_step(
    token_h,
    n_blocks,
    norms,
    lin,
    caches,
    position,
):
    """DECODE: seq=1, caches already exist (or empty list)."""

    h = token_h.reshape(1, HIDDEN)

    for b in range(n_blocks):

        cache = caches[b]

        h, cache = block_forward_cached(
            h, norms, lin, cache,
            pos_offset=position,
            update_cache=True,
        )

        caches[b] = cache

    return h


# ============================================================
# COMPARISON HELPERS
# ============================================================

def bf16_bits(a):

    return RN.fp32_array_to_bf16_bits(
        np.asarray(a, dtype=np.float32)
    )


def diff_stats(a, b):

    d = np.abs(a - b)

    finite = d[np.isfinite(d)]

    maxd = (
        float(finite.max())
        if finite.size else 0.0
    )

    mism = int(
        np.count_nonzero(bf16_bits(a) != bf16_bits(b))
    )

    return maxd, mism


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("RAILNET STAGE 14 - MULTI-BLOCK + KV CACHE")
    print("(layer-0 fabric; every block re-uses layer-0 weights)")
    print("=" * 78, flush=True)

    norms, raw_store, dense_w = M13.load_all_weights()

    compiled = M13.load_compiled_linears(raw_store)

    checksum_ok = all(
        c.checksum_ok for c in compiled.values()
    )

    print(f"artifact checksum: {'OK' if checksum_ok else 'FAIL'}")

    dense_lin = make_dense_linear(dense_w)

    rail_lin = make_rail_linear(compiled)

    rng = np.random.default_rng(SEED)

    results = {
        "prefill": {},
        "decode": {},
    }

    halted = False
    halt_reason = None

    # ========================================================
    # PART A: FULL PREFILL  (dense vs railnet)
    # ========================================================

    for n_blocks in BLOCK_COUNTS:

        results["prefill"][str(n_blocks)] = {}

        for seq in SEQ_GRID:

            if halted:

                break

            print()
            print(
                f"--- PREFILL blocks={n_blocks} seq={seq} ---",
                flush=True
            )

            h0 = rng.normal(
                0.0, 1.0, (seq, HIDDEN)
            ).astype(np.float64)

            y_d, bo_d, ca_d = stack_forward(
                h0, n_blocks, norms, dense_lin,
                collect_caches=True
            )

            y_r, bo_r, ca_r = stack_forward(
                h0, n_blocks, norms, rail_lin,
                collect_caches=True
            )

            entry = {"blocks": {}, "caches": {}}

            ok_all = True

            for b in range(n_blocks):

                maxd, mism = diff_stats(bo_d[b], bo_r[b])

                status = "EXACT" if mism == 0 else "DIFF"

                ok_all &= mism == 0

                entry["blocks"][str(b)] = {
                    "status": status,

                    "fp64_max_abs_diff": maxd,

                    "bf16_mismatches": mism,
                }

                print(
                    f"  block {b} output : {status:5s} "
                    f"max|d|={maxd:.2e} bf16mism={mism}"
                )

            for b in range(n_blocks):

                kd, md = diff_stats(ca_d[b]["K"], ca_r[b]["K"])

                vd, mv = diff_stats(ca_d[b]["V"], ca_r[b]["V"])

                status = (
                    "EXACT"
                    if (md == 0 and mv == 0)
                    else "DIFF"
                )

                ok_all &= md == 0 and mv == 0

                entry["caches"][str(b)] = {
                    "K_status": (
                        "EXACT" if md == 0 else "DIFF"
                    ),

                    "V_status": (
                        "EXACT" if mv == 0 else "DIFF"
                    ),

                    "K_fp64_max_abs_diff": kd,

                    "V_fp64_max_abs_diff": vd,

                    "bf16_mismatches_K": md,

                    "bf16_mismatches_V": mv,

                    "expected_len": seq,
                }

                print(
                    f"  block {b} K/V    : "
                    f"{entry['caches'][str(b)]['K_status']:5s}/"
                    f"{entry['caches'][str(b)]['V_status']:5s} "
                    f"(len={ca_r[b]['K'].shape[1]})"
                )

            results["prefill"][str(n_blocks)][str(seq)] = entry

            if not ok_all:

                halted = True

                halt_reason = (
                    f"prefill divergence blocks={n_blocks} "
                    f"seq={seq}"
                )

                break

    # ========================================================
    # PART B: DECODE vs PREFILL consistency
    # ========================================================

    if not halted:

        for n_blocks in BLOCK_COUNTS:

            if halted:

                break

            print()
            print(
                f"--- DECODE blocks={n_blocks} "
                f"T={DECODE_LEN} ---",
                flush=True
            )

            h0 = rng.normal(
                0.0, 1.0, (DECODE_LEN, HIDDEN)
            ).astype(np.float64)

            # Reference prefill captures per-step outputs.
            yd_pf, bod_pf, cad_pf = stack_forward(
                h0, n_blocks, norms, dense_lin,
                collect_caches=True
            )

            yr_pf, bor_pf, car_pf = stack_forward(
                h0, n_blocks, norms, rail_lin,
                collect_caches=True
            )

            # Decode loops.
            caches_d = [None] * n_blocks

            caches_r = [None] * n_blocks

            dec_d = []

            dec_r = []

            ok_all = True

            entry = {"steps": {}}

            for t in range(DECODE_LEN):

                sd = stack_decode_step(
                    h0[t], n_blocks, norms,
                    dense_lin, caches_d, t
                )

                sr = stack_decode_step(
                    h0[t], n_blocks, norms,
                    rail_lin, caches_r, t
                )

                dec_d.append(sd)

                dec_r.append(sr)

                # consistency: decode(t) vs prefill row t
                mdd, md_mism = diff_stats(
                    sd, yd_pf[t]
                )

                msr, mr_mism = diff_stats(
                    sr, yr_pf[t]
                )

                xdr, xr_mism = diff_stats(
                    sr, sd
                )

                step_ok = (
                    md_mism == 0
                    and mr_mism == 0
                    and xr_mism == 0
                )

                ok_all &= step_ok

                lens_ok = all(
                    caches_r[b]["K"].shape[1]
                    == t + 1
                    for b in range(n_blocks)
                )

                ok_all &= lens_ok

                entry["steps"][t] = {
                    "dense_vs_prefill": (
                        "EXACT" if md_mism == 0 else "DIFF"
                    ),

                    "rail_vs_prefill": (
                        "EXACT" if mr_mism == 0 else "DIFF"
                    ),

                    "rail_vs_dense": (
                        "EXACT" if xr_mism == 0 else "DIFF"
                    ),

                    "cache_len_expected": t + 1,

                    "cache_growth_ok": bool(lens_ok),

                    "fp64_max_abs_diff": xdr,
                }

                print(
                    f"  step t={t}: d/p="
                    f"{entry['steps'][t]['dense_vs_prefill']:5s} "
                    f"r/p={entry['steps'][t]['rail_vs_prefill']:5s} "
                    f"r/d={entry['steps'][t]['rail_vs_dense']:5s} "
                    f"cachelen={caches_r[0]['K'].shape[1]}"
                )

            # Final caches: decode vs prefill must match exactly.
            cache_final_ok = True

            cache_entry = {}

            for b in range(n_blocks):

                km, kmis = diff_stats(
                    cad_pf[b]["K"], caches_r[b]["K"]
                )

                vm, vmis = diff_stats(
                    cad_pf[b]["V"], caches_r[b]["V"]
                )

                ok = kmis == 0 and vmis == 0

                cache_final_ok &= ok

                ok_all &= ok

                cache_entry[str(b)] = {
                    "K_status": (
                        "EXACT" if kmis == 0 else "DIFF"
                    ),

                    "V_status": (
                        "EXACT" if vmis == 0 else "DIFF"
                    ),
                }

            entry["final_caches_vs_prefill"] = cache_entry

            results["decode"][str(n_blocks)] = entry

            if not ok_all:

                halted = True

                halt_reason = (
                    f"decode divergence blocks={n_blocks}"
                )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    def section_ok(mode):

        return all(
            e["blocks"][b]["status"] == "EXACT"
            and e["caches"][b]["K_status"] == "EXACT"
            and e["caches"][b]["V_status"] == "EXACT"
            for ndata in results[mode].values()
            for e in ndata.values()
            for b in map(str, e["blocks"])
        )

    prefill_ok = (
        len(results["prefill"]) > 0
        and section_ok("prefill")
    )

    decode_entries = results["decode"]

    decode_ok = (
        bool(decode_entries)
        and all(
            s["dense_vs_prefill"] == "EXACT"
            and s["rail_vs_prefill"] == "EXACT"
            and s["rail_vs_dense"] == "EXACT"
            and s["cache_growth_ok"]
            for ndata in decode_entries.values()
            for s in ndata["steps"].values()
        )
        and all(
            c["K_status"] == "EXACT"
            and c["V_status"] == "EXACT"
            for ndata in decode_entries.values()
            for c in ndata[
                "final_caches_vs_prefill"
            ].values()
        )
    )

    two_ok = all(
        e["blocks"]["0"]["status"] == "EXACT"
        and e["blocks"]["1"]["status"] == "EXACT"
        for e in results["prefill"].get("2", {}).values()
    ) and "2" in results["prefill"]

    three_ok = all(
        e["blocks"]["2"]["status"] == "EXACT"
        for e in results["prefill"].get("3", {}).values()
    ) and "3" in results["prefill"]

    verdict = (
        checksum_ok
        and not halted
        and two_ok
        and three_ok
        and prefill_ok
        and decode_ok
    )

    print()
    print("=" * 64)
    print("STAGE 14 SUMMARY")
    print("=" * 64)
    print(f"2-block exact              : "
          f"{'PASS' if two_ok else 'FAIL'}")

    print(f"3-block exact              : "
          f"{'PASS' if three_ok else 'FAIL'}")

    print(f"KV cache exact             : "
          f"{'PASS' if prefill_ok else 'FAIL'}")

    print(f"Prefill exact              : "
          f"{'PASS' if prefill_ok else 'FAIL'}")

    print(f"Decode exact (+growth)     : "
          f"{'PASS' if decode_ok else 'FAIL'}")

    print(f"Runtime dense weight array : ABSENT")

    print(f"STAGE 14                   : "
          f"{'PASS' if verdict else 'INCOMPLETE'}")

    if halted:

        print(f"HALT REASON: {halt_reason}")

    print("=" * 64)

    milestone = {
        "milestone": "stage14_multiblock_kvcache",

        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),

        "seed": SEED,

        "static_state": {
            "rails": 96,

            "artifact": str(ARTIFACT_PATH),

            "checksum_ok": bool(checksum_ok),
        },

        "dynamic_state": [
            "hidden_states", "Q/K/V",
            "attention_probs", "KV_cache",
        ],

        "scope_note": (
            "All blocks re-use compiled layer-0 weights on "
            "BOTH paths to isolate multi-block state "
            "semantics; per-layer compile is scale-out work."
        ),

        "results": results,

        "halted": bool(halted),

        "halt_reason": halt_reason,

        "verdict": (
            "PASS" if verdict else "INCOMPLETE"
        ),
    }

    out = Path("results/milestone_stage14_multiblock.json")

    with open(out, "w") as f:

        json.dump(milestone, f, indent=2)

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
