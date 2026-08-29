"""Model-level compile orchestration.

Compiles every dense linear weight of a safetensors model into a RailNet
artifact directory:

    <out_dir>/
      manifest.json
      layers/layer_00/q_proj.json          # rails + routing table (+ sha256)
      layers/layer_00/q_proj.routeids.npy  # per-element route-id map (uint16)
      ...

Norm vectors and the (tied) embedding are NOT compiled — they are read back
from the source safetensors at runtime (spec: compression not claimed for
those). Their names are recorded in ``manifest["constants"]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from railnet.artifacts.manifest import checksum_manifest
from railnet.compiler.compiler import RailNetCompiler
from railnet.safetensors_reader import list_tensors, read_tensor_raw, tensor_metadata

ATTN_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_ROLES = ("gate_proj", "up_proj", "down_proj")


def _classify(name: str):
    """Return (layer_index, role) for a compilable linear, else None."""
    if not name.startswith("model.layers."):
        return None
    parts = name.split(".")
    try:
        layer = int(parts[2])
    except (IndexError, ValueError):
        return None
    sub = ".".join(parts[3:])
    for role in ATTN_ROLES:
        if sub == f"self_attn.{role}.weight":
            return layer, role
    for role in MLP_ROLES:
        if sub == f"mlp.{role}.weight":
            return layer, role
    return None


def _write_artifact(path: Path, tensor) -> str:
    """Write a single compiled-tensor JSON with a canonical sha256, return the digest."""
    data = tensor.to_dict()
    data["shape"] = list(tensor.shape.dims)
    data["checksum_sha256"] = checksum_manifest(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return data["checksum_sha256"]


def compile_model(
    safetensors_path: str,
    out_dir: str = "compiled",
    dtype: str = "bf16",
    rails: int = 96,
    max_terms: int = 4,
    max_iters: int = 300,
    only: str | None = None,
    limit: int | None = None,
    config_path: str | None = None,
    tokenizer_path: str | None = None,
    verbose: bool = True,
) -> dict:
    src = Path(safetensors_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if config_path is None:
        guess = src.parent / "config.json"
        config_path = str(guess) if guess.exists() else None
    config = json.loads(Path(config_path).read_text()) if config_path else {}

    if tokenizer_path is None:
        guess = src.parent / "tokenizer.json"
        tokenizer_path = str(guess) if guess.exists() else None

    compiler = RailNetCompiler(model="generic", default_dtype=dtype)

    targets = []
    for name in list_tensors(model_file=src):
        hit = _classify(name)
        if hit is None:
            continue
        if only and only not in name:
            continue
        targets.append((name, *hit))
    targets.sort(key=lambda t: (t[1], t[2]))
    if limit is not None:
        targets = targets[:limit]

    manifest: dict = {
        "model": config.get("model_type", "generic"),
        "dtype": dtype,
        "config": config,
        "source_model": str(src),
        "tokenizer": tokenizer_path,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "compiler": {"rails": rails, "max_terms": max_terms},
        "runtime_weight_array": "ABSENT",
        "tensors": {},
        "constants": {
            "norm_suffixes": [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ],
            "final_norm": "model.norm.weight",
            "embedding": "model.embed_tokens.weight",
            "lm_head": "tied_to_embedding",
            "strategy": "read from source_model at runtime (not compiled)",
        },
    }

    failed = []
    for i, (name, layer, role) in enumerate(targets):
        raw, shape = read_tensor_raw(name, model_file=src)
        meta = tensor_metadata(name, model_file=src)
        assert meta["dtype"] == "BF16", f"{name}: expected BF16, got {meta['dtype']}"
        if verbose:
            print(f"[{i + 1}/{len(targets)}] {name}  shape={tuple(shape)}", flush=True)
        try:
            tensor = compiler.compile_tensor(
                raw,
                dtype=dtype,
                rails=rails,
                max_terms=max_terms,
                name=name,
                shape=tuple(shape),
                max_iters=max_iters,
            )
        except RuntimeError as exc:
            failed.append((name, str(exc)))
            manifest["tensors"][name] = {"status": "FAILED", "error": str(exc)}
            if verbose:
                print(f"    FAILED: {exc}", flush=True)
            continue

        art_path = out / "layers" / f"layer_{layer:02d}" / f"{role}.json"
        map_path = out / "layers" / f"layer_{layer:02d}" / f"{role}.routeids.npy"
        digest = _write_artifact(art_path, tensor)
        assert tensor.route_ids is not None
        np.save(map_path, tensor.route_ids.astype(np.uint16).reshape(tuple(shape)))

        manifest["tensors"][name] = {
            "status": "PASS",
            "layer": layer,
            "role": role,
            "shape": list(shape),
            "artifact": str(art_path.relative_to(out)),
            "route_map": str(map_path.relative_to(out)),
            "sha256": digest,
        }

    manifest["pass_count"] = sum(
        1 for e in manifest["tensors"].values() if e.get("status") == "PASS"
    )
    manifest["fail_count"] = len(failed)
    manifest["verdict"] = "PASS" if not failed and targets else "INCOMPLETE"
    manifest["checksum_sha256"] = checksum_manifest(manifest)

    man_path = out / "manifest.json"
    tmp = man_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(man_path)

    if verbose:
        print(
            f"\n{manifest['pass_count']}/{len(targets)} tensors compiled "
            f"-> {man_path}  verdict={manifest['verdict']}",
            flush=True,
        )
    return manifest
