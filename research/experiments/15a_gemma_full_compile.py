import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# RAILNET STAGE 15A - FULL MODEL COMPILATION (GEMMA3 1B)
#
# Streaming / mmap-only tensor access, resumable via manifest,
# atomic artifacts with SHA-256, per-tensor exact validation.
#
# Default fabric: 96 rails / 4 terms (Layer-0-validated).
# Escalation ladder on failure: 96 -> 128 -> 192.
# Norm vectors: runtime constants (not compiled; tiny).
# Embedding:    exact mmap row-lookup strategy (NOT compressed;
#               compression NOT CLAIMED - spec 50).
# LM head:      tied to embedding (verified: absent from file,
#               HF _tied_weights_keys) -> same strategy note.
#
# CLI:
#   --resume            skip PASS tensors with valid checksums
#   --limit N           compile at most N new tensors this run
#   --minutes M         graceful time budget then save+exit
#   --only SUBSTR       compile tensors containing substring
#   --layers a,b        restrict layer indices
# ============================================================


HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

from railnet import compiler as C          # noqa: E402

from railnet import artifact as ART        # noqa: E402

from railnet import safetensors_reader as SR   # noqa: E402


CONFIG_FILE = HERE / "model_data/config.json"

MANIFEST_PATH = HERE / "compiled/manifest.json"

COMPILED_DIR = HERE / "compiled"


def classify(name):

    if name.startswith("model.embed_tokens"):

        return "embedding", None, "lookup_exact"

    if name.startswith("lm_head"):

        return "lm_head", None, "tied_to_embedding"

    parts = name.split(".")

    if not name.startswith("model.layers."):

        if name.startswith("model.norm"):

            return "norm", None, "runtime_constant"

        return "other", None, "unhandled"

    li = int(parts[2])

    sub = ".".join(parts[3:])

    if sub.endswith("_norm.weight"):

        return "norm", li, "runtime_constant"

    for tag in ("q_proj", "k_proj",
                "v_proj", "o_proj"):

        if sub == f"self_attn.{tag}.weight":

            return "attention_linear", li, tag

    for tag in ("gate_proj", "up_proj", "down_proj"):

        if sub == f"mlp.{tag}.weight":

            return "mlp_linear", li, tag

    return "other", li, "unhandled"


def scan_model():

    entries = []

    for name in SR.list_tensors():

        cat, li, role = classify(name)

        meta = SR.tensor_metadata(name)

        entries.append(
            {
                "name": name,

                "category": cat,

                "layer": li,

                "role": role,

                "shape": list(meta["shape"]),

                "dtype": meta["dtype"],

                "parameters": int(
                    np.prod(meta["shape"])
                ),
            }
        )

    return entries


def load_manifest():

    if MANIFEST_PATH.exists():

        with open(MANIFEST_PATH, encoding="utf-8") as f:

            return json.load(f)

    manifest = {
        "model": {

            "path": str(
                SR.MODEL_FILE.relative_to(HERE)
            ),
        },

        "compiler": {
            "rails_default": C.DEFAULT_RAILS,

            "terms_default": C.TERMS,

            "ladder": C.LADDER,

            "max_iters": C.RN.MAX_ITERS,
        },

        "tensors": {},
    }

    return manifest


def save_manifest(m):

    MANIFEST_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    tmp = MANIFEST_PATH.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:

        json.dump(m, f, indent=2)

    import os

    os.replace(tmp, MANIFEST_PATH)


def already_done(manifest, name):

    e = manifest["tensors"].get(name)

    if not e or e.get("status") != "PASS":

        return False

    art = e.get("artifact")

    if not art or not Path(art).exists():

        return False

    ok, _ = ART.verify_checksum(art)

    return ok


def compile_one(entry, manifest, log):

    name = entry["name"]

    raw, shape = SR.read_tensor_raw(name)

    result = C.compile_tensor_lossless(
        raw, name, log=log
    )

    if result["status"] != "PASS":

        manifest["tensors"][name] = {
            "status": "FAILED",

            "category": entry["category"],

            "layer": entry["layer"],

            "shape": entry["shape"],

            "unique": result["unique"],

            "attempts": result["attempts"],
        }

        return False

    li = entry["layer"]

    short = entry["role"]

    out_dir = (
        COMPILED_DIR
        / "layers"
        / f"layer_{li:02d}"
    )

    art_path = out_dir / f"{short}_lossless.json"

    content = ART.build_lossless_artifact(
        name,
        shape,
        result["rails_arr"],
        result["terms"],
        result["table"],
        result["exact"],
        result["unique"],
        attempts=result["attempts"],
    )

    ART.save_artifact_atomic(art_path, content)

    map_path = ART.save_route_map_atomic(
        out_dir / f"{short}.routeids.npy",
        raw,
    )

    ok_ck, _ = ART.verify_checksum(art_path)

    manifest["tensors"][name] = {
        "status": (
            "PASS" if ok_ck else "CHECKSUM_FAIL"
        ),

        "category": entry["category"],

        "layer": li,

        "shape": entry["shape"],

        "rails": result["rails"],

        "terms": result["terms"],

        "exact": bool(ok_ck),

        "unique_values": result["unique"],

        "artifact": str(art_path.relative_to(HERE)),

        "route_map": str(
            Path(map_path).relative_to(HERE)
        ),

        "sha256": ART.sha256_file(art_path),

        "attempts": result["attempts"],
    }

    return ok_ck


def main():

    args = sys.argv[1:]

    resume = "--resume" in args

    limit = None

    minutes = None

    only = None

    layers_filter = None

    def val(flag):

        return args[args.index(flag) + 1]

    if "--limit" in args:

        limit = int(val("--limit"))

    if "--minutes" in args:

        minutes = float(val("--minutes"))

    if "--only" in args:

        only = val("--only")

    if "--layers" in args:

        layers_filter = set(
            int(x)
            for x in val("--layers").split(",")
        )

    print("=" * 78)
    print("RAILNET STAGE 15A - FULL MODEL COMPILER")
    print("=" * 78)

    cfg = json.load(open(CONFIG_FILE))

    n_layers = cfg["num_hidden_layers"]

    entries = scan_model()

    linear_entries = [
        e for e in entries
        if e["category"] in (
            "attention_linear", "mlp_linear"
        )
    ]

    if layers_filter is not None:

        linear_entries = [
            e for e in linear_entries
            if e["layer"] in layers_filter
        ]

    if only:

        linear_entries = [
            e for e in linear_entries
            if only in e["name"]
        ]

    linear_entries.sort(
        key=lambda e: (
            e["layer"],
            ["q_proj", "k_proj", "v_proj",
             "o_proj", "gate_proj", "up_proj",
             "down_proj"].index(e["role"]),
        )
    )

    total_params = sum(
        e["parameters"] for e in linear_entries
    )

    print(f"linear tensors : {len(linear_entries)}")
    print(f"linear params  : {total_params:,}")
    print(f"default fabric : "
          f"{C.DEFAULT_RAILS} rails / {C.TERMS} terms")

    manifest = load_manifest()

    done_before = sum(
        1 for e in linear_entries
        if manifest["tensors"].get(e["name"], {}).get(
            "status"
        ) == "PASS"
    )

    print(f"manifest       : {done_before} already PASS")
    print(f"resume         : {resume}")
    print(flush=True)

    compiled_now = 0

    failed_now = []

    started = time.perf_counter()

    stop_reason = "QUEUE_COMPLETE"

    for i, entry in enumerate(linear_entries):

        name = entry["name"]

        if (
            resume
            and already_done(manifest, name)
        ):

            continue

        if (
            limit is not None
            and compiled_now >= limit
        ):

            stop_reason = "LIMIT_REACHED"

            break

        if (
            minutes is not None
            and (
                time.perf_counter() - started
            ) / 60.0 >= minutes
        ):

            stop_reason = "TIME_BUDGET_REACHED"

            break

        print(
            f"[{i + 1}/{len(linear_entries)}] "
            f"{name}",
            flush=True
        )

        try:

            ok = compile_one(
                entry, manifest,
                log=lambda s: print("   " + s, flush=True)
            )

        except Exception as exc:

            ok = False

            manifest["tensors"][name] = {
                "status": "ERROR",

                "error": repr(exc),
            }

        if ok:

            compiled_now += 1

        else:

            failed_now.append(name)

        # Persist manifest after EVERY tensor (spec 40/41).
        save_manifest(manifest)

    # ---- non-linear registry ------------------------------

    manifest["non_compiled_registry"] = {}

    for e in entries:

        if e["category"] in (
            "attention_linear", "mlp_linear"
        ):

            continue

        manifest["non_compiled_registry"][e["name"]] = {
            "category": e["category"],

            "strategy": e["role"],
        }

    save_manifest(manifest)

    passed_total = sum(
        1 for e in linear_entries
        if manifest["tensors"].get(e["name"], {}).get(
            "status"
        ) == "PASS"
    )

    print()
    print("=" * 78)
    print("STAGE 15A SESSION SUMMARY")
    print("=" * 78)
    print(f"stop reason        : {stop_reason}")
    print(f"compiled this run  : {compiled_now}")
    print(f"failed this run    : {len(failed_now)}")
    print(
        f"total PASS         : {passed_total}/"
        f"{len(linear_entries)}"
    )

    if failed_now:

        print("failed tensors:")
        for n in failed_now:
            print(f"  {n}")

    report = {
        "phase": "15a",

        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),

        "stop_reason": stop_reason,

        "session_compiled": compiled_now,

        "session_failed": failed_now,

        "total_pass": passed_total,

        "total_linear": len(linear_entries),

        "verdict_phase_a": (
            "PASS"
            if passed_total == len(linear_entries)
            else "INCOMPLETE"
        ),
    }

    outdir = HERE / "results/stage15"

    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "compile_report.json", "w") as f:

        json.dump(report, f, indent=2)

    print(f"\nSaved: results/stage15/compile_report.json")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
