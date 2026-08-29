"""Independent cross-check: RailNet vs HuggingFace ``transformers`` Gemma3.

This promotes the RailNet claim from "rail path == our dense reference" to
"the whole graph matches a reference implementation". It needs the real
weights and ``pip install 'railnet[hf]'``.

    git lfs pull
    python research/reproduce_gemma.py --limit 0            # or a full compile
    python research/crosscheck_hf.py --prompt "The capital of France is"

RailNet runs in float64; HF runs in bfloat16 (or float32 with --hf-fp32), so
the logits are NOT expected to be bitwise identical. The bar here is
behavioural: same argmax, high top-k overlap, small logit delta, and an
identical greedy continuation.

Writes results/crosscheck_hf.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from railnet.runtime.transformer import RailNetModel

ROOT = Path(__file__).resolve().parent.parent


def _hf_logits(model_dir: str, ids: list[int], fp32: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if fp32 else torch.bfloat16
    try:
        hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    except TypeError:  # older transformers
        hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    hf.eval()
    with torch.no_grad():
        out = hf(torch.tensor([ids]))
    return out.logits[0, -1].float().cpu().numpy(), hf


def _hf_greedy(hf, ids: list[int], n: int) -> list[int]:
    import torch

    cur = list(ids)
    with torch.no_grad():
        for _ in range(n):
            logits = hf(torch.tensor([cur])).logits[0, -1]
            cur.append(int(torch.argmax(logits)))
    return cur[len(ids) :]


def _compare(a: np.ndarray, b: np.ndarray, k: int = 5) -> dict:
    ta = set(np.argsort(a)[-k:].tolist())
    tb = set(np.argsort(b)[-k:].tolist())
    return {
        "argmax_match": int(np.argmax(a)) == int(np.argmax(b)),
        "argmax_railnet": int(np.argmax(a)),
        "argmax_reference": int(np.argmax(b)),
        f"top{k}_overlap": len(ta & tb),
        "max_abs_delta": float(np.max(np.abs(a - b))),
        "mean_abs_delta": float(np.mean(np.abs(a - b))),
        "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", default=str(ROOT / "compiled"))
    ap.add_argument("--model-dir", default=str(ROOT / "model_data"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--greedy-tokens", type=int, default=8)
    ap.add_argument("--hf-fp32", action="store_true", help="run HF in float32 (closer to RailNet)")
    args = ap.parse_args()

    if (Path(args.compiled) / "manifest.json").exists():
        model = RailNetModel.load(args.compiled)
        have_compile = model.is_fully_compiled
        if not have_compile:
            print("compiled/ is incomplete — dense-only cross-check (graph fidelity)")
    else:
        model = RailNetModel.from_source(Path(args.model_dir) / "model.safetensors")
        have_compile = False
        print("no compiled/ — dense-only cross-check (graph fidelity)")

    tok = model.get_tokenizer()
    ids = list(tok.encode(args.prompt).ids)

    hf_last, hf = _hf_logits(args.model_dir, ids, args.hf_fp32)
    dense_last = model.forward(ids, backend="dense")
    dense_greedy = model.generate(
        ids, max_new_tokens=args.greedy_tokens, tokenizer=tok, backend="dense"
    )["tokens"]
    hf_greedy = _hf_greedy(hf, ids, args.greedy_tokens)

    report: dict = {
        "prompt": args.prompt,
        "prompt_tokens": ids,
        "hf_dtype": "float32" if args.hf_fp32 else "bfloat16",
        "compiled": have_compile,
        "railnet_dense_vs_hf": _compare(dense_last, hf_last),
        "greedy_railnet_dense": dense_greedy,
        "greedy_hf": hf_greedy,
        "greedy_dense_match": dense_greedy == hf_greedy,
        "greedy_text_hf": tok.decode(hf_greedy),
    }
    if have_compile:
        rail_last = model.forward(ids, backend="rail")
        rail_greedy = model.generate(ids, max_new_tokens=args.greedy_tokens, tokenizer=tok)[
            "tokens"
        ]
        report["railnet_rail_vs_hf"] = _compare(rail_last, hf_last)
        report["greedy_railnet_rail"] = rail_greedy
        report["greedy_rail_match"] = rail_greedy == hf_greedy
        report["greedy_text_railnet"] = tok.decode(rail_greedy)

    report["verdict"] = (
        "PASS"
        if report["railnet_dense_vs_hf"]["argmax_match"] and report["greedy_dense_match"]
        else "REVIEW"
    )

    out = ROOT / "results" / "crosscheck_hf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
