"""RailNet CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_inspect(args):
    from railnet.models import get_adapter

    adapter = get_adapter(args.model)
    info = adapter.inspect(args.path)
    print(json.dumps(info, indent=2))


def cmd_compile(args):
    from railnet.compiler.model import compile_model

    manifest = compile_model(
        args.path,
        out_dir=args.out,
        dtype=args.dtype,
        rails=args.rails,
        max_terms=args.terms,
        max_iters=args.max_iters,
        only=args.only,
        limit=args.limit,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "verdict": manifest["verdict"],
                "pass_count": manifest["pass_count"],
                "fail_count": manifest["fail_count"],
                "out_dir": args.out,
            },
            indent=2,
        )
    )


def cmd_verify(args):
    from railnet.artifacts import verify_checksum, verify_rnmodel

    p = Path(args.path)
    if p.suffix == ".rnmodel":
        ok, info = verify_rnmodel(str(p))
        print(json.dumps({"ok": ok, "header": info}, indent=2))
    elif p.is_dir() or p.name == "manifest.json":
        from railnet.compiler.model import verify_compiled

        print(json.dumps(verify_compiled(str(p.parent if p.is_file() else p)), indent=2))
    else:
        ok, _data = verify_checksum(str(p))
        print(json.dumps({"ok": ok}, indent=2))


def cmd_run(args):
    from railnet.runtime.transformer import RailNetModel

    model = RailNetModel.load(args.artifact)
    logits = model.forward([int(t) for t in args.tokens.split(",")])
    import numpy as np

    print(json.dumps({"argmax": int(np.argmax(logits)), "vocab": int(logits.shape[0])}, indent=2))


def cmd_generate(args):
    from railnet.runtime.transformer import RailNetModel

    model = RailNetModel.load(args.artifact)
    out = model.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_benchmark(args):
    print("Benchmark: use benchmarks/ scripts + verification overlays. See SPEC.md")


def cmd_info(args):
    from railnet.artifacts.reader import verify_rnmodel

    _ok, info = verify_rnmodel(args.path) if Path(args.path).suffix == ".rnmodel" else (False, {})
    print(json.dumps(info, indent=2))


def build_parser():
    p = argparse.ArgumentParser(prog="railnet")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect")
    s.add_argument("path")
    s.add_argument("--model", default="gemma3")
    s.set_defaults(func=cmd_inspect)
    s = sub.add_parser("compile")
    s.add_argument("path")
    s.add_argument("--model", default="gemma3")
    s.add_argument("--dtype", default="bf16")
    s.add_argument("--rails", type=int, default=96)
    s.add_argument("--terms", type=int, default=4)
    s.add_argument("--out", default="compiled")
    s.add_argument("--max-iters", type=int, default=300, help="basis-learning iteration cap")
    s.add_argument("--only", default=None, help="substring filter on tensor names")
    s.add_argument("--limit", type=int, default=None, help="compile at most N tensors")
    s.add_argument("--resume", action="store_true", help="skip tensors already in --out")
    s.set_defaults(func=cmd_compile)
    s = sub.add_parser("verify")
    s.add_argument("path")
    s.set_defaults(func=cmd_verify)
    s = sub.add_parser("run")
    s.add_argument("artifact")
    s.add_argument("--tokens", default="1,2,3", help="comma-separated input token ids")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("generate")
    s.add_argument("artifact")
    s.add_argument("--prompt", default="Hello")
    s.add_argument("--max-tokens", type=int, default=32)
    s.set_defaults(func=cmd_generate)
    s = sub.add_parser("benchmark")
    s.add_argument("model", nargs="?")
    s.set_defaults(func=cmd_benchmark)
    s = sub.add_parser("artifact")
    s.add_argument("sub")
    s.add_argument("path")
    s.set_defaults(func=cmd_info)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
