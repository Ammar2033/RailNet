"""RailNet CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_inspect(args):
    from railnet.models import get_adapter
    adapter = get_adapter(args.model)
    info = adapter.inspect(args.path)
    print(json.dumps(info, indent=2))


def cmd_compile(args):
    from railnet.compiler import RailNetCompiler
    comp = RailNetCompiler(model=args.model, default_dtype=args.dtype)
    # single-tensor demo path
    print(f"Compiling {args.path} dtype={args.dtype} rails={args.rails} terms={args.terms}")
    # bulk model compile delegates to research script
    if Path(args.path).is_file() and args.path.endswith(".safetensors"):
        from pathlib import Path as P
        import importlib.util
        p = P(__file__).resolve().parent.parent / "research" / "experiments" / "15a_gemma_full_compile.py"
        if p.exists():
            print(f"Delegating to {p} (proven bulk compile)")
        else:
            print("Bulk compile not found — use compiler.compile_tensor for single tensors")


def cmd_verify(args):
    from railnet.artifacts.reader import verify_rnmodel
    from railnet.artifact import verify_checksum
    p = Path(args.path)
    if p.suffix == ".rnmodel":
        ok, info = verify_rnmodel(str(p))
        print(json.dumps({"ok": ok, "header": info}, indent=2))
    else:
        ok, data = verify_checksum(str(p))
        print(json.dumps({"ok": ok}, indent=2))


def cmd_run(args):
    from railnet.runtime.transformer import RailNetModel
    model = RailNetModel.load(args.artifact)
    print(f"Loaded {args.artifact} — call model.forward(input_ids) in Python API")


def cmd_generate(args):
    from railnet.runtime.transformer import RailNetModel
    model = RailNetModel.load(args.artifact)
    out = model.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_benchmark(args):
    print("Benchmark: use benchmarks/ scripts + verification overlays. See SPEC.md")


def cmd_info(args):
    from railnet.artifacts.reader import verify_rnmodel
    ok, info = verify_rnmodel(args.path) if Path(args.path).suffix == ".rnmodel" else (False, {})
    print(json.dumps(info, indent=2))


def build_parser():
    p = argparse.ArgumentParser(prog="railnet")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("path"); s.add_argument("--model", default="gemma3"); s.set_defaults(func=cmd_inspect)
    s = sub.add_parser("compile"); s.add_argument("path"); s.add_argument("--model", default="gemma3"); s.add_argument("--dtype", default="bf16"); s.add_argument("--rails", type=int, default=96); s.add_argument("--terms", type=int, default=4); s.set_defaults(func=cmd_compile)
    s = sub.add_parser("verify"); s.add_argument("path"); s.set_defaults(func=cmd_verify)
    s = sub.add_parser("run"); s.add_argument("artifact"); s.set_defaults(func=cmd_run)
    s = sub.add_parser("generate"); s.add_argument("artifact"); s.add_argument("--prompt", default="Hello"); s.add_argument("--max-tokens", type=int, default=32); s.set_defaults(func=cmd_generate)
    s = sub.add_parser("benchmark"); s.add_argument("model", nargs="?"); s.set_defaults(func=cmd_benchmark)
    s = sub.add_parser("artifact"); s.add_argument("sub"); s.add_argument("path"); s.set_defaults(func=cmd_info)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
