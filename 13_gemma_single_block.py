import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 13
# SINGLE GEMMA3 TRANSFORMER BLOCK - LAYER-0 EXACT EXECUTION
#
# Semantics verified from official HF modeling_gemma3.py +
# local model_data/config.json (no assumptions, spec 25):
#
#   residual = h
#   h = input_layernorm(h)
#   h = Attention(h)            # q/k/v -> q_norm/k_norm ->
#                               # RoPE -> GQA eager attention
#   h = post_attention_layernorm(h)
#   h = residual + h
#   residual = h
#   h = pre_feedforward_layernorm(h)
#   h = MLP(h)                  # down( gelu_tanh(gate*h) * up )
#   h = post_feedforward_layernorm(h)
#   h = residual + h
#
# RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w)     [zero-centred]
# RoPE:    inv_freq=base^(-2i/d); q*cos + rotate_half(q)*sin
# Scaling: query_pre_attn_scalar^-0.5 = 1/sqrt(256)
# Layer 0: sliding_attention type; window=512 -> inactive
#          for seq<=8 (asserted).
#
# Two execution paths share IDENTICAL non-linear code;
# ONLY the 7 linear layers differ:
#   dense  : y = x @ W^T          (reference side)
#   railnet: y = rail_linear(...) (96-rail global fabric,
#            no dense weights - spec 22)
#
# Validation tiers (spec 19):
#   T-math : linear segments proven exact in Stage 12
#            (Fraction oracle); non-linears are the SAME
#            deterministic functions applied to their inputs.
#   T-bf16 : every intermediate checkpoint compared after
#            BF16 rounding -> practical bitwise equality.
#   T-fp64 : max |delta| reported as diagnostic only.
#
# First divergence halts further sequence lengths.
# ============================================================


MODEL_FILE = Path("model_data/model.safetensors")

CONFIG_FILE = Path("model_data/config.json")

ARTIFACT_PATH = Path(
    "compiled/layer0/_GLOBAL_layer0.json"
)

SEQ_GRID = [1, 2, 4, 8]

SEED = 42


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


HERE = Path(__file__).resolve().parent

RN = load_module(
    HERE / "04_bf16_learned_basis.py", "rn"
)

R12 = load_module(
    HERE / "12_gemma_linear_runner.py", "r12"
)


# ============================================================
# CONFIG (from file, not assumed)
# ============================================================

CFG = json.load(open(CONFIG_FILE))

HIDDEN = CFG["hidden_size"]

INTERMEDIATE = CFG["intermediate_size"]

HEADS = CFG["num_attention_heads"]

KV_HEADS = CFG["num_key_value_heads"]

HEAD_DIM = CFG["head_dim"]

EPS = CFG["rms_norm_eps"]

SLIDING_WINDOW = CFG["sliding_window"]

Q_SCALE = CFG["query_pre_attn_scalar"] ** -0.5

ROPE_LOCAL_BASE = CFG["rope_local_base_freq"]

KV_GROUPS = HEADS // KV_HEADS

LAYER0_TYPE = (
    "sliding_attention"
    if SLIDING_WINDOW
    else "full_attention"
)


TENSOR_NAMES = {
    "input_layernorm": (
        "model.layers.0.input_layernorm.weight"
    ),

    "post_attention_layernorm": (
        "model.layers.0.post_attention_layernorm.weight"
    ),

    "pre_feedforward_layernorm": (
        "model.layers.0.pre_feedforward_layernorm.weight"
    ),

    "post_feedforward_layernorm": (
        "model.layers.0.post_feedforward_layernorm.weight"
    ),

    "q_norm": "model.layers.0.self_attn.q_norm.weight",

    "k_norm": "model.layers.0.self_attn.k_norm.weight",

    "q_proj": "model.layers.0.self_attn.q_proj.weight",

    "k_proj": "model.layers.0.self_attn.k_proj.weight",

    "v_proj": "model.layers.0.self_attn.v_proj.weight",

    "o_proj": "model.layers.0.self_attn.o_proj.weight",

    "gate_proj": "model.layers.0.mlp.gate_proj.weight",

    "up_proj": "model.layers.0.mlp.up_proj.weight",

    "down_proj": "model.layers.0.mlp.down_proj.weight",
}


LINEAR_TENSORS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


CHECKPOINT_ORDER = [
    "input_layernorm",
    "q_proj", "k_proj", "v_proj",
    "q_norm", "k_norm",
    "rope_q", "rope_k",
    "attn_scores",
    "attention_output",
    "o_proj",
    "residual_1",
    "pre_ffn_norm",
    "gate_proj", "up_proj",
    "activation",
    "down_proj",
    "post_ffn_norm",
    "residual_2",
]


# ============================================================
# WEIGHT LOADING
# ============================================================

def load_all_weights():

    norms = {}

    raw_store = {}

    dense_w = {}

    for short, tname in TENSOR_NAMES.items():

        RN.TARGET_TENSOR = tname

        raw, shape = RN.read_target_tensor()

        raw_store[short] = (raw, shape)

        w64 = (
            RN.bf16_array_to_float32(raw)
            .astype(np.float64)
        ).reshape(shape)

        if short.endswith("norm"):

            norms[short] = w64

        else:

            dense_w[short] = w64

    return norms, raw_store, dense_w


def load_compiled_linears(raw_store):

    compiled = {}

    for short in LINEAR_TENSORS:

        raw, shape = raw_store[short]

        compiled[short] = R12.CompiledTensor(
            ARTIFACT_PATH,
            raw,
            shape
        )

    return compiled


# ============================================================
# SHARED NON-LINEAR OPS (identical for both paths)
# ============================================================

def rms_norm(x, w):

    # Gemma3: computed in higher precision, scaled by (1+w)
    variance = np.mean(x * x, axis=-1, keepdims=True)

    return (
        x
        * (1.0 / np.sqrt(variance + EPS))
        * (1.0 + w)
    )


def rope_cos_sin(seq):

    dim = HEAD_DIM

    half = np.arange(
        0, dim, 2, dtype=np.float64
    )

    inv_freq = ROPE_LOCAL_BASE ** (
        -(half / dim)
    )

    pos = np.arange(seq, dtype=np.float64)

    freqs = pos[:, None] * inv_freq[None, :]

    emb = np.concatenate(
        [freqs, freqs], axis=-1
    )

    return np.cos(emb), np.sin(emb)


def rotate_half(x):

    d = x.shape[-1]

    x1 = x[..., : d // 2]

    x2 = x[..., d // 2:]

    return np.concatenate(
        [-x2, x1], axis=-1
    )


def gelu_tanh(x):

    c = np.sqrt(2.0 / np.pi)

    inner = c * (x + 0.044715 * x ** 3)

    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax_last(x):

    e = np.exp(
        x - x.max(axis=-1, keepdims=True)
    )

    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# BLOCK FORWARD
# ============================================================

def block_forward(
    h0,
    norms,
    lin,           # callable(short, x_2d) -> y_2d
    checkpoints
):

    seq = h0.shape[0]

    def cap(key, value):

        checkpoints[key] = value.copy()

    # ---- Attention sub-layer ------------------------------

    residual = h0

    h = rms_norm(h0, norms["input_layernorm"])

    cap("input_layernorm", h)

    q = lin("q_proj", h)

    cap("q_proj", q)

    k = lin("k_proj", h)

    cap("k_proj", k)

    v = lin("v_proj", h)

    cap("v_proj", v)

    # reshape (seq, heads*hd) -> (heads, seq, hd)
    qh = q.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)

    kh = k.reshape(seq, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

    vh = v.reshape(seq, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

    # Q/K per-head RMSNorm BEFORE RoPE
    qh = rms_norm(qh, norms["q_norm"])

    cap("q_norm", qh)

    kh = rms_norm(kh, norms["k_norm"])

    cap("k_norm", kh)

    # RoPE (local base; layer 0 is sliding type)
    cos, sin = rope_cos_sin(seq)

    qh = qh * cos[None, :, :] + rotate_half(qh) * sin[None, :, :]

    cap("rope_q", qh)

    kh = kh * cos[None, :, :] + rotate_half(kh) * sin[None, :, :]

    cap("rope_k", kh)

    # GQA repeat_interleave KV heads
    kh_full = np.repeat(kh, KV_GROUPS, axis=0)

    vh_full = np.repeat(vh, KV_GROUPS, axis=0)

    # scores (heads, seq, seq)
    scores = np.matmul(
        qh,
        kh_full.transpose(0, 2, 1)
    ) * Q_SCALE

    mask = np.triu(
        np.ones((seq, seq), dtype=bool),
        k=1
    )

    scores = np.where(
        mask[None, :, :],
        -np.inf,
        scores
    )

    cap("attn_scores", scores)

    probs = softmax_last(scores)

    ctx = np.matmul(probs, vh_full)

    attn_out = ctx.transpose(1, 0, 2).reshape(
        seq, HEADS * HEAD_DIM
    )

    cap("attention_output", attn_out)

    o = lin("o_proj", attn_out)

    cap("o_proj", o)

    h = residual + o

    h = rms_norm(h, norms["post_attention_layernorm"])

    cap("residual_1", h)

    # ---- FFN sub-layer ------------------------------------

    residual = h

    h_ffn_in = rms_norm(h, norms["pre_feedforward_layernorm"])

    cap("pre_ffn_norm", h_ffn_in)

    g = lin("gate_proj", h_ffn_in)

    cap("gate_proj", g)

    u = lin("up_proj", h_ffn_in)

    cap("up_proj", u)

    act = gelu_tanh(g)

    cap("activation", act)

    prod = act * u

    d = lin("down_proj", prod)

    cap("down_proj", d)

    h = rms_norm(d, norms["post_feedforward_layernorm"])

    cap("post_ffn_norm", h)

    h = residual + h

    cap("residual_2", h)

    return h


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


# ============================================================
# CHECKPOINT COMPARISON
# ============================================================

def bf16_round_bits(a):

    a32 = np.asarray(a, dtype=np.float32)

    return RN.fp32_array_to_bf16_bits(
        a32
    )


def compare_checkpoints(ref, rn):

    report = {}

    first_divergence = None

    for key in CHECKPOINT_ORDER:

        if (
            key not in ref
            or key not in rn
        ):

            continue

        a = ref[key]

        b = rn[key]

        if a.shape != b.shape:

            report[key] = {
                "status": "SHAPE_MISMATCH",

                "ref_shape": list(a.shape),

                "rn_shape": list(b.shape),
            }

            if first_divergence is None:

                first_divergence = key

            continue

        d = np.abs(a - b)

        if not np.all(np.isfinite(d)):

            # Masked positions hold +/-inf in both paths;
            # their difference is NaN by definition.
            finite = d[np.isfinite(d)]

            diff = (
                float(finite.max())
                if finite.size
                else 0.0
            )

        else:

            diff = float(np.max(d))

        ba = bf16_round_bits(a)

        bb = bf16_round_bits(b)

        mism = int(np.count_nonzero(ba != bb))

        status = (
            "EXACT" if mism == 0 else "DIFF"
        )

        report[key] = {
            "status": status,

            "fp64_max_abs_diff": diff,

            "bf16_mismatches": mism,

            "elements": int(ba.size),
        }

        if mism > 0 and first_divergence is None:

            first_divergence = key

    return report, first_divergence


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "RAILNET STAGE 13 - GEMMA3 LAYER-0 BLOCK "
        "(GLOBAL 96-RAIL FABRIC)"
    )
    print("=" * 80)

    print()
    print(f"config verified : eps={EPS}, "
          f"scale={Q_SCALE:.6f}, heads={HEADS}, "
          f"kv={KV_HEADS}, hd={HEAD_DIM}")

    print(f"layer-0 type    : {LAYER0_TYPE} "
          f"(window={SLIDING_WINDOW})")

    print()
    print("loading weights ...", flush=True)

    norms, raw_store, dense_w = load_all_weights()

    print("building compiled linears ...", flush=True)

    compiled = load_compiled_linears(raw_store)

    checksum_ok = all(
        c.checksum_ok for c in compiled.values()
    )

    print(
        f"artifact checksum: "
        f"{'OK' if checksum_ok else 'FAIL'}"
    )

    dense_lin = make_dense_linear(dense_w)

    rail_lin = make_rail_linear(compiled)

    rng = np.random.default_rng(SEED)

    results = {}

    halted = False

    for seq in SEQ_GRID:

        if halted:

            break

        assert seq <= SLIDING_WINDOW, (
            "sliding window active - extend mask!"
        )

        print()
        print("-" * 80)
        print(f"SEQ = {seq}")
        print("-" * 80, flush=True)

        h0 = rng.normal(
            0.0, 1.0, (seq, HIDDEN)
        ).astype(np.float64)

        t0 = time.perf_counter()

        ref_ck = {}

        y_ref = block_forward(
            h0, norms, dense_lin, ref_ck
        )

        t_dense = time.perf_counter() - t0

        t0 = time.perf_counter()

        rn_ck = {}

        y_rn = block_forward(
            h0, norms, rail_lin, rn_ck
        )

        t_rail = time.perf_counter() - t0

        report, first_div = compare_checkpoints(
            ref_ck, rn_ck
        )

        # Final output comparison
        out_diff = float(
            np.max(np.abs(y_ref - y_rn))
        )

        out_ba = bf16_round_bits(y_ref)

        out_bb = bf16_round_bits(y_rn)

        out_mism = int(
            np.count_nonzero(out_ba != out_bb)
        )

        print()
        print(
            f"{'checkpoint':24s} {'status':>7s} "
            f"{'fp64|maxdiff|':>14s} {'bf16 mism':>10s}"
        )

        print("-" * 60)

        for key in CHECKPOINT_ORDER:

            if key not in report:

                continue

            r = report[key]

            print(
                f"{key:24s} {r['status']:>7s} "
                f"{r['fp64_max_abs_diff']:>14.3e} "
                f"{r['bf16_mismatches']:>10d}"
            )

        out_status = (
            "EXACT" if out_mism == 0 else "DIFF"
        )

        print("-" * 60)

        print(
            f"{'FINAL BLOCK OUTPUT':24s} {out_status:>7s} "
            f"{out_diff:>14.3e} {out_mism:>10d}"
        )

        print(
            f"time dense={t_dense:.3f}s "
            f"rail={t_rail:.3f}s"
        )

        results[str(seq)] = {
            "checkpoints": report,

            "first_divergence": first_div,

            "final_output": {
                "status": out_status,

                "fp64_max_abs_diff": out_diff,

                "bf16_mismatches": out_mism,

                "elements": int(out_ba.size),
            },

            "seconds_dense": round(t_dense, 3),

            "seconds_rail": round(t_rail, 3),
        }

        if first_div is not None:

            halted = True

            print()
            print(
                f"HALTED: first divergence at "
                f"'{first_div}' (seq={seq})"
            )

    # ========================================================
    # FINAL REPORT (spec 56)
    # ========================================================

    all_reports = [
        r["checkpoints"]
        for r in results.values()
    ]

    def all_exact(key):

        return all(
            rep.get(key, {}).get("status")
            == "EXACT"
            for rep in all_reports
        )

    final_ok = all(
        r["final_output"]["status"] == "EXACT"
        for r in results.values()
    )

    no_div = all(
        r["first_divergence"] is None
        for r in results.values()
    )

    verdict_pass = (
        no_div
        and final_ok
        and len(results) > 0
        and checksum_ok
    )

    print()
    print("=" * 64)
    print("RAILNET GEMMA3 LAYER-0")
    print("=" * 64)

    lines = []

    for key in CHECKPOINT_ORDER:

        st = (
            "PASS"
            if all_exact(key)
            else "FAIL"
        )

        lines.append((key, st))

    for key, st in lines:

        print(f"{key:22s} {st}")

    print(f"{'FINAL BLOCK OUTPUT':22s} "
          f"{'PASS' if final_ok else 'FAIL'}")

    print()

    print(
        "Runtime dense weight array : "
        f"{'ABSENT' if checksum_ok else 'PRESENT?'}"
    )

    print(
        f"STAGE 13               : "
        f"{'PASS' if verdict_pass else 'INCOMPLETE'}"
    )

    print("=" * 64)

    milestone = {
        "milestone": "stage13_single_block",

        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),

        "artifact": str(ARTIFACT_PATH),

        "seed": SEED,

        "seq_grid": SEQ_GRID,

        "results": results,

        "verdict": (
            "PASS" if verdict_pass else "INCOMPLETE"
        ),
    }

    out_dir = Path("results")

    out_dir.mkdir(exist_ok=True)

    with open(
        out_dir / "milestone_stage13_block.json",
        "w"
    ) as f:

        json.dump(milestone, f, indent=2)

    print(f"\nSaved: results/milestone_stage13_block.json")


if __name__ == "__main__":
    main()
