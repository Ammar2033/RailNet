import ctypes
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 15B - FULL GEMMA3 FORWARD (26 LAYERS)
#
# token IDs
#   -> embedding exact row lookup (mmap; NOT compressed)
#   -> 26 decoder layers via RailNet linear kernels
#   -> final RMSNorm
#   -> tied LM head (chunked over vocab, mmap)
#   -> logits
#
# Runtime linear-weight policy:
#   NO dense linear weight arrays are loaded. Each layer's
#   rails/topology/route-map artifacts are streamed in and
#   released layer-by-layer. Embedding/LM head use exact
#   mmap row access (disclosed separately, spec 50/51).
#
# Validation:
#   - per-layer boundary: bf16-bit equality + fp64 diagnostic
#   - FIRST DIVERGENCE halts with spec-47 style detail
#   - last-token logits: max|d|, bf16 mism, top-1/top-5
#   - memory + compute audits
# ============================================================


HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

from railnet import kernel as K          # noqa: E402

from railnet import safetensors_reader as SR  # noqa: E402

from railnet import transformer as T     # noqa: E402

from railnet import validation as V      # noqa: E402

from railnet.embedding import MmapRowLookup  # noqa: E402


MANIFEST = HERE / "compiled/manifest.json"

CONFIG_FILE = HERE / "model_data/config.json"

RESULTS = HERE / "results/stage15"

SEQ_GRID_DEFAULT = [1, 2, 4, 8]

SEED = 42

LINEARS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

NORMS = [
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "q_norm",
    "k_norm",
]


def rss_bytes():

    try:

        import psutil  # optional

        return psutil.Process().memory_info().rss

    except Exception:

        pass

    try:

        class PMC(ctypes.Structure):

            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()

        pmc.cb = ctypes.sizeof(PMC)

        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc),
            pmc.cb,
        )

        return int(pmc.WorkingSetSize)

    except Exception:

        return -1


def load_config():

    cfg = json.load(open(CONFIG_FILE))

    T.init_from_config(cfg)

    return cfg


def bf16_to_f64(raw):

    return (
        raw.astype(np.uint32) << 16
    ).view(np.float32).astype(np.float64)


def load_layer_norms(layer_idx):

    norms = {}

    for key in NORMS:

        name = (
            f"model.layers.{layer_idx}."
            f"self_attn.{key}.weight"
        ) if key in ("q_norm", "k_norm") else (
            f"model.layers.{layer_idx}."
            f"{key}.weight"
        )

        raw, _shape = SR.read_tensor_raw(name)

        norms[key] = bf16_to_f64(raw)

    return norms


def load_final_norm():

    raw, _shape = SR.read_tensor_raw("model.norm.weight")

    return bf16_to_f64(raw)


def load_compiled_layer(layer_idx):

    """Build rail-linear backend for one layer (streamed)."""

    compiled = {}

    for short in LINEARS:

        name = (
            f"model.layers.{layer_idx}."
            f"self_attn.{short}.weight"
        ) if short in (
            "q_proj", "k_proj", "v_proj", "o_proj"
        ) else f"model.layers.{layer_idx}.mlp.{short}.weight"

        entry = MANIFEST_DATA["tensors"][name]

        art_path = HERE / entry["artifact"]

        map_path = HERE / entry["route_map"]

        route_ids = np.load(map_path)

        shape = tuple(entry["shape"])

        compiled[short] = K.CompiledTensor(
            art_path, route_ids, shape
        )

    def lin(short, x):

        out = np.empty(
            (x.shape[0], compiled[short].out_features),
            dtype=np.float64,
        )

        for r in range(x.shape[0]):

            out[r] = K.rail_linear(
                x[r].astype(np.float64),
                compiled[short],
            )

        return out

    return lin, compiled


def make_dense_backend(layer_idx):

    dense_w = {}

    for short in LINEARS:

        name = (
            f"model.layers.{layer_idx}."
            f"self_attn.{short}.weight"
        ) if short in (
            "q_proj", "k_proj", "v_proj", "o_proj"
        ) else f"model.layers.{layer_idx}.mlp.{short}.weight"

        raw, shape = SR.read_tensor_raw(name)

        dense_w[short] = bf16_to_f64(raw).reshape(shape)

    def lin(short, x):

        return x @ dense_w[short].T

    return lin, dense_w


def run_stack(h, n_layers, backend_factory):

    caches = [None] * n_layers

    boundaries = []

    for b in range(n_layers):

        norms_b = load_layer_norms(b)

        lin, _keep = backend_factory(b)

        h, caches[b] = T.block_forward(
            h, norms_b, lin, cache=caches[b],
            pos_offset=0,
        )

        boundaries.append(h.copy())

    return h, boundaries, caches


def main():

    global MANIFEST_DATA

    args = sys.argv[1:]

    seqs = SEQ_GRID_DEFAULT

    if "--seqs" in args:

        seqs = [
            int(x)
            for x in args[args.index("--seqs") + 1].split(",")
        ]

    print("=" * 78)
    print("RAILNET STAGE 15B - FULL GEMMA3 FORWARD (26 LAYERS)")
    print("=" * 78)

    cfg = load_config()

    n_layers = cfg["num_hidden_layers"]

    hidden = cfg["hidden_size"]

    vocab = cfg["vocab_size"]

    manifest_ok = MANIFEST.exists()

    assert manifest_ok, "run 15a first"

    MANIFEST_DATA = json.load(open(MANIFEST))

    pass_count = sum(
        1 for e in MANIFEST_DATA["tensors"].values()
        if e.get("status") == "PASS"
    )

    print(f"manifest PASS tensors : {pass_count}/182")

    assert pass_count == 182, "Phase A incomplete"

    emb = MmapRowLookup()

    print(
        f"embedding strategy    : exact mmap row lookup "
        f"(NOT compressed; compression NOT CLAIMED)"
    )

    print(f"lm_head               : tied to embedding "
          f"(verified absent from file); chunked exact matmul")

    rng = np.random.default_rng(SEED)

    mem0 = rss_bytes()

    results = {}

    halted_seq = None

    for seq in seqs:

        print()
        print("-" * 78)
        print(f"SEQ = {seq}")
        print("-" * 78, flush=True)

        token_ids = rng.integers(
            1000, vocab - 1000, size=seq
        )

        h_emb_rail = emb.rows_f64(token_ids)

        # Reference embedding uses the same EXACT rows.
        h_emb_ref = h_emb_rail.copy()

        t0 = time.perf_counter()

        h_ref, bounds_ref, _c = run_stack(
            h_emb_ref.copy(), n_layers,
            lambda b: make_dense_backend(b),
        )

        t_dense = time.perf_counter() - t0

        t0 = time.perf_counter()

        h_rn, bounds_rn, caches_rn = run_stack(
            h_emb_rail.copy(), n_layers,
            lambda b: load_compiled_layer(b),
        )

        t_rail = time.perf_counter() - t0

        # ---- per-layer boundary checks ---------------------

        layer_rows = []

        first_div = None

        for b in range(n_layers):

            maxd, mism = V.diff_stats(
                bounds_ref[b], bounds_rn[b]
            )

            ok = mism == 0

            if not ok and first_div is None:

                detail = V.first_divergence_detail(
                    bounds_ref[b], bounds_rn[b],
                    extra={
                        "layer": b,

                        "checkpoint":
                            "block_output",
                    },
                )

                first_div = {
                    "layer": b,
                    "detail": detail,
                }

            layer_rows.append(
                {
                    "layer": b,

                    "status": (
                        "EXACT" if ok else "DIFF"
                    ),

                    "fp64_max_abs_diff": maxd,

                    "bf16_mismatches": mism,

                    "elements": int(
                        bounds_ref[b].size
                    ),
                }
            )

            status_char = "." if ok else "X"

            print(
                f"  L{b:02d} {status_char} "
                f"max|d|={maxd:.2e} "
                f"bf16mism={mism}",
                flush=True
            )

        # ---- final norm + logits ---------------------------

        fn_w = load_final_norm()

        hf_ref = T.rms_norm(h_ref[-1:], fn_w)

        hf_rn = T.rms_norm(h_rn[-1:], fn_w)

        fmaxd, fmism = V.diff_stats(hf_ref, hf_rn)

        logits_ref = emb.logits_chunked(hf_ref[0])

        logits_rn = emb.logits_chunked(hf_rn[0])

        lmaxd, lmism = V.diff_stats(logits_ref, logits_rn)

        top1_same = int(
            np.argmax(logits_ref) == np.argmax(logits_rn)
        )

        top5_same = int(
            set(
                np.argsort(logits_ref)[-5:].tolist()
            )
            == set(
                np.argsort(logits_rn)[-5:].tolist()
            )
        )

        print(
            f"  final norm : max|d|={fmaxd:.2e} "
            f"bf16mism={fmism}"
        )

        print(
            f"  logits     : max|d|={lmaxd:.2e} "
            f"bf16mism={lmism}/{vocab} "
            f"top1same={top1_same} top5same={top5_same}"
        )

        seq_pass = (
            first_div is None
            and fmism == 0
            and lmism == 0
        )

        results[str(seq)] = {
            "layers": layer_rows,

            "first_divergence": first_div,

            "final_norm": {
                "status": (
                    "EXACT" if fmism == 0 else "DIFF"
                ),

                "fp64_max_abs_diff": fmaxd,
            },

            "logits": {
                "status": (
                    "EXACT" if lmism == 0 else "DIFF"
                ),

                "fp64_max_abs_diff": lmaxd,

                "bf16_mismatches": lmism,

                "top1_same": bool(top1_same),

                "top5_same": bool(top5_same),

                "note": (
                    "tied LM head over identical exact "
                    "embedding rows"
                ),
            },

            "seconds_reference": round(t_dense, 2),

            "seconds_railnet": round(t_rail, 2),

            "pass": bool(seq_pass),
        }

        if not seq_pass:

            halted_seq = seq

            break

    mem1 = rss_bytes()

    # ---- compute audit (analytic, spec 37) -----------------

    seqs_run = sorted(int(s) for s in results.keys())

    dense_mults = 0

    shared_upper = 0

    for name, e in MANIFEST_DATA["tensors"].items():

        out_f, in_f = e["shape"]

        rails = e.get("rails", 96)

        for s in seqs_run:

            dense_mults += out_f * in_f * s

            shared_upper += out_f * rails * s

    lm_mults = vocab * hidden * sum(seqs_run)

    compute_report = {
        "mode": "ANALYTIC_ESTIMATE",

        "dense_linear_multiplications": dense_mults,

        "railnet_shared_multiplications_upper_bound":
            shared_upper,

        "estimated_reduction": float(
            1.0 - shared_upper / dense_mults
        ),

        "lm_head_multiplications_tied_exact": lm_mults,

        "note": (
            "shared count is an analytic upper bound "
            "(per output x rail pairs); measured Layer-0 "
            "censuses came close to this bound"
        ),
    }

    memory_report = {
        "rss_before_bytes": int(mem0),

        "rss_after_bytes": int(mem1),

        "peak_delta_bytes": int(max(0, mem1 - mem0)),

        "route_map_static_bytes": int(sum(
            (HERE / e["route_map"]).stat().st_size
            for e in MANIFEST_DATA["tensors"].values()
        )),

        "artifact_json_bytes": int(sum(
            (HERE / e["artifact"]).stat().st_size
            for e in MANIFEST_DATA["tensors"].values()
        )),

        "embedding_strategy": (
            "exact mmap row lookup; full matrix NOT resident; "
            "compression NOT CLAIMED"
        ),
    }

    all_pass = bool(results) and all(
        r["pass"] for r in results.values()
    )

    verdict = "PASS" if all_pass else (
        "FAIL" if halted_seq is not None else "INCOMPLETE"
    )

    print()
    print("=" * 64)
    print("RAILNET GEMMA3 STAGE 15B")
    print("=" * 64)

    for s, r in results.items():

        print(
            f"seq={s:2s} : "
            f"{'PASS' if r['pass'] else 'FAIL'} "
            f"(rail {r['seconds_railnet']}s / ref "
            f"{r['seconds_reference']}s)"
        )

    print(f"\nSTAGE 15B VERDICT : {verdict}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    with open(
        RESULTS / "forward_report.json", "w"
    ) as f:

        json.dump(
            {
                "phase": "15b",

                "seq_grid": list(results.keys()),

                "results": results,

                "halted_seq": halted_seq,

                "verdict": verdict,
            },
            f, indent=2,
        )

    with open(
        RESULTS / "memory_report.json", "w"
    ) as f:

        json.dump(memory_report, f, indent=2)

    with open(
        RESULTS / "compute_report.json", "w"
    ) as f:

        json.dump(compute_report, f, indent=2)

    print("Saved: results/stage15/{forward,memory,compute}_report.json")


if __name__ == "__main__":
    main()
