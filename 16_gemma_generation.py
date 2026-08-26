import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 16
# REAL TOKENIZER + DETERMINISTIC GREEDY GENERATION
#
# prompt text -> tokenizer -> RailNet prefill (26 layers,
# KV cache) -> argmax -> decode steps with growing cache ->
# tokenizer decode -> text.
#
# Reference pass runs FIRST with dense backend, records
# tokens + logit bf16 bits + caches; then the RailNet pass
# regenerates and compares per step. Dense weights are never
# visible to the railnet path.
#
# Greedy only: temperature=0, no sampling (spec 3).
# ============================================================

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

try:

    sys.stdout.reconfigure(encoding="utf-8")

except Exception:

    pass

from railnet import kernel as K              # noqa: E402

from railnet import safetensors_reader as SR  # noqa: E402

from railnet import transformer as T         # noqa: E402

from railnet import validation as V          # noqa: E402

from railnet.embedding import MmapRowLookup  # noqa: E402


MANIFEST = HERE / "compiled/manifest.json"

CONFIG_FILE = HERE / "model_data/config.json"

TOKENIZER_FILE = HERE / "model_data/tokenizer.json"

RESULTS = HERE / "results/stage16"

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

SEED = 42


def rss_bytes():

    try:

        import psutil

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


def bf16_to_f64(raw):

    return (
        raw.astype(np.uint32) << 16
    ).view(np.float32).astype(np.float64)


def load_tokenizer():

    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_FILE))


def load_config():

    cfg = json.load(open(CONFIG_FILE))

    T.init_from_config(cfg)

    return cfg


# ============================================================
# BACKENDS
# ============================================================

class DenseSide:

    """Reference generation. NOTHING resident: weights are
    streamed per layer from safetensors (Stage-15b profile)."""

    name = "reference"

    def __init__(self, n_layers):

        self.n_layers = n_layers

        self.norms = [
            _load_layer_norms(b)
            for b in range(n_layers)
        ]

    def layer_backend(self, b):

        def lin(short, x):

            base = (
                f"model.layers.{b}.self_attn.{short}.weight"
            ) if short in (
                "q_proj", "k_proj", "v_proj", "o_proj"
            ) else (
                f"model.layers.{b}.mlp.{short}.weight"
            )

            raw, shape = SR.read_tensor_raw(base)

            w = bf16_to_f64(raw).reshape(shape)

            return x @ w.T

        return lin

    def release(self):

        import gc

        gc.collect()


class RailSide:

    """RailNet generation. Memory profile mirrors proven
    Stage-15b: route maps IN RAM (mmap causes page-fault
    storms on 16GB machines), original kernel, no prepare()."""

    name = "railnet"

    def __init__(
        self, n_layers, manifest_data,
    ):

        self.n_layers = n_layers

        self.norms = [
            _load_layer_norms(b)
            for b in range(n_layers)
        ]

        self.layers = []

        for b in range(n_layers):

            compiled = {}

            for short in LINEARS:

                base = (
                    f"model.layers.{b}.self_attn.{short}.weight"
                ) if short in (
                    "q_proj", "k_proj", "v_proj", "o_proj"
                ) else (
                    f"model.layers.{b}.mlp.{short}.weight"
                )

                e = manifest_data["tensors"][base]

                # IN-RAM load: memmap indexing caused hard
                # page-fault storms on this machine.
                route_ids = np.load(HERE / e["route_map"])

                c = K.CompiledTensor(
                    HERE / e["artifact"],
                    route_ids,
                    tuple(e["shape"]),
                )

                compiled[short] = c

            self.layers.append(compiled)

            if (b + 1) % 4 == 0 or b + 1 == n_layers:

                print(
                    f"   [rail-load] layers "
                    f"{b + 1}/{n_layers}",
                    flush=True,
                )

    def layer_backend(self, b):

        comp = self.layers[b]

        def lin(short, x):

            out = np.empty(
                (x.shape[0], comp[short].out_features),
                dtype=np.float64,
            )

            for r in range(x.shape[0]):

                out[r] = K.rail_linear(
                    x[r].astype(np.float64),
                    comp[short],
                )

            return out

        return lin


def _load_layer_norms(b):

    norms = {}

    for key in NORMS:

        name = (
            f"model.layers.{b}.self_attn.{key}.weight"
        ) if key in ("q_norm", "k_norm") else (
            f"model.layers.{b}.{key}.weight"
        )

        raw, _shape = SR.read_tensor_raw(name)

        norms[key] = bf16_to_f64(raw)

    return norms


def generate(
    side,
    emb,
    fn_w,
    eos_ids,
    token_ids,
    max_new_tokens,
):
    """
    Greedy generation for one backend.

    Returns dict with tokens, per-step logits bits,
    final caches, timings.
    """

    caches = [None] * side.n_layers

    h = emb.rows_f64(token_ids)

    t_pref0 = time.perf_counter()

    for b in range(side.n_layers):

        h, caches[b] = T.block_forward(
            h, side.norms[b],
            side.layer_backend(b),
            cache=caches[b],
            pos_offset=0,
        )

    prefill_s = time.perf_counter() - t_pref0

    print(
        f"   [prefill done: {prefill_s:.1f}s]",
        flush=True,
    )

    hf = T.rms_norm(h[-1:], fn_w)

    logits = emb.logits_chunked(hf[0])

    next_tok = int(np.argmax(logits))

    tokens = [next_tok]

    logit_bits = [V.bf16_bits(logits)]

    gen_t0 = time.perf_counter()

    step_times = []

    pos = len(token_ids)

    while (
        len(tokens) < max_new_tokens
        and next_tok not in eos_ids
    ):

        s0 = time.perf_counter()

        h = emb.rows_f64([next_tok])

        for b in range(side.n_layers):

            h, caches[b] = T.block_forward(
                h, side.norms[b],
                side.layer_backend(b),
                cache=caches[b],
                pos_offset=pos,
            )

        hf = T.rms_norm(h[-1:], fn_w)

        logits = emb.logits_chunked(hf[0])

        next_tok = int(np.argmax(logits))

        tokens.append(next_tok)

        logit_bits.append(V.bf16_bits(logits))

        step_times.append(time.perf_counter() - s0)

        pos += 1

        print(
            f"   [decode {len(tokens)}/{max_new_tokens}] "
            f"{step_times[-1]:.1f}s tok={next_tok}",
            flush=True,
        )

    total_gen_s = time.perf_counter() - gen_t0

    return {
        "tokens": tokens,

        "logit_bits": logit_bits,

        "caches": caches,

        "cache_len": int(
            caches[-1]["K"].shape[1]
        ),

        "prefill_seconds": round(prefill_s, 2),

        "decode_seconds": round(total_gen_s, 2),

        "mean_step_seconds": round(
            float(np.mean(step_times)), 2
        )
        if step_times else 0.0,
    }


# ============================================================
# MAIN
# ============================================================

def run_prompt(
    prompt,
    max_new,
    tok,
    cfg,
    emb,
    fn_w,
    eos_ids,
    n_layers,
    manifest_data,
):

    print()
    print("=" * 78)
    print(f"PROMPT: {prompt!r}")
    print("=" * 78, flush=True)

    enc = tok.encode(prompt)

    prompt_ids = enc.ids

    print(
        f"prompt tokens ({len(prompt_ids)}): "
        f"{prompt_ids[:16]}{'...' if len(prompt_ids)>16 else ''}"
    )

    print("REFERENCE generation ...", flush=True)

    print(
        f"  [rss before ref: {rss_bytes()/1048576:.0f} MB]",
        flush=True
    )

    dense = DenseSide(n_layers)

    ref = generate(
        dense, emb, fn_w, eos_ids,
        prompt_ids, max_new,
    )

    dense.release()

    import gc

    gc.collect()

    try:

        ctypes.windll.psapi.SetProcessWorkingSetSize(
            ctypes.windll.kernel32.GetCurrentProcess(),
            -1, -1,
        )

    except Exception:

        pass

    print(
        f"  [rss after ref trim: "
        f"{rss_bytes()/1048576:.0f} MB]",
        flush=True
    )

    ref_text = tok.decode(ref["tokens"])

    print(
        f"  ref tokens ({len(ref['tokens'])}): "
        f"{ref['tokens'][:20]}"
    )

    print(f"  ref text : {ref_text[:160]!r}")

    print("RAILNET generation ...", flush=True)

    print(
        f"  [rss before rail: {rss_bytes()/1048576:.0f} MB]",
        flush=True
    )

    rail = RailSide(n_layers, manifest_data)

    rn = generate(
        rail, emb, fn_w, eos_ids,
        prompt_ids, max_new,
    )

    del rail

    rn_text = tok.decode(rn["tokens"])

    print(
        f"  rnt tokens ({len(rn['tokens'])}): "
        f"{rn['tokens'][:20]}"
    )

    print(f"  rnt text : {rn_text[:160]!r}")

    # ---- comparisons --------------------------------------

    seq_exact = ref["tokens"] == rn["tokens"]

    first_div_step = None

    logit_mism_total = 0

    step0_mism = None

    n_steps = min(
        len(ref["logit_bits"]),
        len(rn["logit_bits"]),
    )

    for s in range(n_steps):

        rb = ref["logit_bits"][s]

        nb = rn["logit_bits"][s]

        mism = int(np.count_nonzero(rb != nb))

        if s == 0:

            step0_mism = mism

        logit_mism_total += mism

        if mism and first_div_step is None:

            first_div_step = s

    kv_ok = True

    kv_stats = []

    for b in range(n_layers):

        km, _ = V.diff_stats(
            ref["caches"][b]["K"],
            rn["caches"][b]["K"],
        )

        vm, _ = V.diff_stats(
            ref["caches"][b]["V"],
            rn["caches"][b]["V"],
        )

        kmis = int(np.count_nonzero(
            V.bf16_bits(ref["caches"][b]["K"])
            != V.bf16_bits(rn["caches"][b]["K"])
        ))

        vmis = int(np.count_nonzero(
            V.bf16_bits(ref["caches"][b]["V"])
            != V.bf16_bits(rn["caches"][b]["V"])
        ))

        ok = kmis == 0 and vmis == 0

        kv_ok &= ok

        kv_stats.append(
            {"layer": b,
             "K_exact": kmis == 0,
             "V_exact": vmis == 0,
             "len":
                 int(rn["caches"][b]["K"].shape[1])}
        )

    text_exact = ref_text == rn_text

    passed = (
        seq_exact
        and logit_mism_total == 0
        and kv_ok
    )

    print()
    print(
        f"  TOKEN SEQUENCE EXACT : {seq_exact}"
    )

    print(
        f"  LOGIT BF16 MISMATCH  : {logit_mism_total} "
        f"(over {n_steps} steps x vocab)"
    )

    print(
        f"  KV CACHE EXACT       : {kv_ok}"
    )

    print(
        f"  TEXT OUTPUT EXACT    : {text_exact}"
    )

    print(
        f"  RESULT               : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "prompt": prompt,

        "prompt_tokens": prompt_ids,

        "max_new_tokens": max_new,

        "reference_tokens": ref["tokens"],

        "railnet_tokens": rn["tokens"],

        "reference_text": ref_text,

        "railnet_text": rn_text,

        "token_sequence_exact": bool(seq_exact),

        "text_exact": bool(text_exact),

        "logit_bf16_mismatch_total":
            logit_mism_total,

        "first_divergence_step": first_div_step,

        "kv_cache_exact": bool(kv_ok),

        "kv_layers": kv_stats,

        "cache_len_final": rn["cache_len"],

        "runtime_dense_weights": False,

        # Step 0 logits come from pure prefill.
        "prefill_exact": bool(step0_mism == 0),

        "prefill_step0_logit_mismatches": step0_mism,

        "timings": {
            "ref_prefill_s":
                ref["prefill_seconds"],

            "ref_decode_s":
                ref["decode_seconds"],

            "ref_ms_per_token": round(
                ref["mean_step_seconds"] * 1000, 1
            ),

            "rail_prefill_s":
                rn["prefill_seconds"],

            "rail_decode_s":
                rn["decode_seconds"],

            "rail_ms_per_token": round(
                rn["mean_step_seconds"] * 1000, 1
            ),
        },

        "pass": bool(passed),
    }


def main():

    args = sys.argv[1:]

    def val(flag, default=None):

        return (
            args[args.index(flag) + 1]
            if flag in args else default
        )

    prompt = val("--prompt")

    max_new = int(val("--max-new-tokens", "8"))

    suite = "--suite" in args

    interactive = "--interactive" in args

    print("=" * 78)
    print("RAILNET STAGE 16 - DETERMINISTIC GENERATION")
    print("=" * 78)

    tok = load_tokenizer()

    cfg = load_config()

    n_layers = cfg["num_hidden_layers"]

    eos_ids = set()

    for eid in cfg.get("eos_token_id", []):

        eos_ids.add(int(eid))

    print(
        f"tokenizer vocab   : {tok.get_vocab_size()}"
    )

    print(f"eos ids           : {sorted(eos_ids)}")

    # ---- tokenizer round-trip report -----------------------

    rt_prompts = [
        "Hello",
        "Explain what an operating system is.",
        "Write a small Python function.",
        "Merhaba dünya.",
        "Selam, bugün yapay zekâ hakkında konuşalım.",
        "Hello\nworld",
        "2 + 2 =",
    ]

    rt_report = []

    rt_ok = True

    for p in rt_prompts:

        e = tok.encode(p)

        d = tok.decode(e.ids)

        stripped_ok = (
            d.strip() == p.strip()
            or d == p
        )

        rt_ok &= stripped_ok

        rt_report.append(
            {
                "prompt": p,

                "tokens": e.ids[:24],

                "n_tokens": len(e.ids),

                "decoded": d,

                "roundtrip_ok": bool(stripped_ok),
            }
        )

        print(
            f"[rt] {p!r:50s} -> "
            f"{len(e.ids):3d} tok  ok={stripped_ok}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)

    with open(
        RESULTS / "tokenizer_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "library": "tokenizers",

                "vocab_size": tok.get_vocab_size(),

                "tests": rt_report,

                "pass": bool(rt_ok),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    emb = MmapRowLookup()

    fn_w = bf16_to_f64(
        SR.read_tensor_raw("model.norm.weight")[0]
    )

    manifest_data = json.load(open(MANIFEST))

    mem0 = rss_bytes()

    tests = []

    if prompt:

        tests.append((prompt, max_new))

    if suite:

        suite_cases = [
            ("Hello, my name is", 10),

            ("Explain what a computer is in simple words.", 5),

            ("2 + 2 =", 10),

            ("Merhaba dünya.", 5),
        ]

        tests.extend(suite_cases)

    if interactive:

        while True:

            try:

                p = input("\nPrompt (>quit): ").strip()

            except EOFError:

                break

            if not p or p.lower() == "quit":

                break

            r = run_prompt(
                p, max_new, tok, cfg, emb, fn_w,
                eos_ids, n_layers, manifest_data,
            )

            print("\nRailNet:", r["railnet_text"])

    else:

        all_pass = bool(tests) and rt_ok

        for i, (p, m) in enumerate(tests):

            r = run_prompt(
                p, m, tok, cfg, emb, fn_w,
                eos_ids, n_layers, manifest_data,
            )

            r["test_index"] = i

            tests_path = (
                RESULTS
                / f"generation_test_{i:03d}.json"
            )

            with open(
                tests_path, "w", encoding="utf-8"
            ) as f:

                json.dump(
                    r, f, indent=2,
                    ensure_ascii=False,
                )

            all_pass &= r["pass"]

        verdict = (
            "PASS"
            if all_pass else "FAIL/INCOMPLETE"
        )

        mem1 = rss_bytes()

        summary = {
            "stage": 16,

            "tokenizer_roundtrip_ok": bool(rt_ok),

            "tests_run": len(tests),

            "all_pass": bool(all_pass),

            "verdict": verdict,

            "rss_delta_mb": round(
                max(0, mem1 - mem0) / 1048576, 1
            ),

            "note": (
                "greedy decoding; correctness-first CPU "
                "simulator; performance secondary (spec 34)"
            ),
        }

        with open(
            RESULTS / "stage16_summary.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary, f, indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 64)
        print("STAGE 16 SUMMARY")
        print("=" * 64)
        print(f"tokenizer roundtrip : "
              f"{'PASS' if rt_ok else 'FAIL'}")
        print(f"generation tests    : "
              f"{len(tests)}")
        print(f"ALL PASS            : {all_pass}")
        print(f"STAGE 16 VERDICT    : {verdict}")
        print("=" * 64)


def results_dir_list():

    return list(
        (HERE / "results/stage16").glob(
            "generation_test_*.json"
        )
    )


if __name__ == "__main__":
    main()
