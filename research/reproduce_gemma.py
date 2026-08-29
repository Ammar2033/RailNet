"""Reproduce the Gemma RailNet result with the current package API.

    python research/reproduce_gemma.py --model model_data/model.safetensors \
        --out compiled --prompt "Hello" --max-new-tokens 8

Steps:
  1. compile_model  -> compiled/ (resumable; use --limit / --only while iterating)
  2. verify_compiled -> structural check of the artifact directory
  3. verify_forward  -> RailNet rail path vs the dense reference (same graph),
     BF16-bitwise, full vocab + per layer
  4. verify_generation -> greedy tokens must match between the two backends

What this proves: the rail representation + kernel are LOSSLESS relative to the
dense computation of the same transformer graph. Token-for-token equivalence
with HuggingFace Gemma3 is a separate validation item (see docs/EXACTNESS.md).

Writes results/gemma_repro.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railnet.compiler.model import compile_model, verify_compiled
from railnet.runtime.transformer import RailNetModel
from railnet.verification import verify_forward, verify_generation

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "model_data/model.safetensors"))
    ap.add_argument("--out", default=str(ROOT / "compiled"))
    ap.add_argument("--rails", type=int, default=96)
    ap.add_argument("--terms", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=300)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip tensors already compiled in --out")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument(
        "--lean", action="store_true", help="memory-tight: stream refs, non-prepared kernel"
    )
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.model)
    if src.stat().st_size < 1_000_000:
        raise SystemExit(
            f"{src} is {src.stat().st_size} bytes — looks like a git-lfs pointer. "
            "Run `git lfs pull` to fetch the weights."
        )

    report: dict = {"model": str(src), "out": args.out}

    if not args.skip_compile:
        m = compile_model(
            str(src),
            out_dir=args.out,
            rails=args.rails,
            max_terms=args.terms,
            max_iters=args.max_iters,
            only=args.only,
            limit=args.limit,
            resume=args.resume,
        )
        report["compile"] = {
            "verdict": m["verdict"],
            "pass_count": m["pass_count"],
            "fail_count": m["fail_count"],
        }

    report["verify_compiled"] = verify_compiled(args.out)

    model = RailNetModel.load(args.out)
    model.lean = args.lean
    if not model.is_fully_compiled:
        report["verdict"] = "INCOMPLETE"
        report["note"] = "not every layer is compiled yet — rerun with --resume to finish"
        _write(report)
        return 1

    tok = model.get_tokenizer()
    ids = list(tok.encode(args.prompt).ids)

    report["verify_forward"] = verify_forward(model, ids, per_layer=True)
    report["verify_generation"] = verify_generation(
        model, ids, max_new_tokens=args.max_new_tokens, tokenizer=tok
    )

    report["verdict"] = (
        "PASS"
        if (
            report["verify_compiled"]["ok"]
            and report["verify_forward"]["verdict"] == "PASS"
            and report["verify_generation"]["verdict"] == "PASS"
        )
        else "FAIL"
    )

    _write(report)
    return 0 if report["verdict"] == "PASS" else 1


def _write(report: dict) -> None:
    out = ROOT / "results" / "gemma_repro.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
