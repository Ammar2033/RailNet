"""Whole-model verification: RailNet rail path vs the dense reference.

Both paths share the exact same transformer ops (norms, RoPE, attention,
activation) and the same embedding / final norm streamed from the source
safetensors — only the linear backend differs. So a BF16-bitwise match here
is a direct statement about the rail kernel + routing, spec-style
(dense reference logits vs RailNet logits).
"""

from __future__ import annotations

import numpy as np

from railnet.dtypes.bf16 import fp32_array_to_bf16_bits


def _bits(a: np.ndarray) -> np.ndarray:
    return fp32_array_to_bf16_bits(np.asarray(a, dtype=np.float32))


def verify_forward(model, input_ids, per_layer: bool = True) -> dict:
    """Compare ``model.forward`` (rail) with ``model.forward_dense`` on one prompt."""
    if not getattr(model, "is_fully_compiled", True):
        raise RuntimeError("model is not fully compiled — the rail path needs every layer")
    rail = model.forward(input_ids, backend="rail", capture_hidden=per_layer)
    dense = model.forward(input_ids, backend="dense", capture_hidden=per_layer)
    rail_logits, rail_h = rail if per_layer else (rail, [])
    dense_logits, dense_h = dense if per_layer else (dense, [])

    logit_mismatch = int(np.count_nonzero(_bits(rail_logits) != _bits(dense_logits)))
    result: dict = {
        "prompt_len": len(list(np.asarray(input_ids).reshape(-1))),
        "vocab": int(rail_logits.shape[0]),
        "logit_bf16_mismatch": logit_mismatch,
        "logits_exact": logit_mismatch == 0,
        "runtime_dense_weights": False,
    }
    if per_layer:
        layers = []
        first_div = None
        for b, (rh, dh) in enumerate(zip(rail_h, dense_h, strict=True)):
            mm = int(np.count_nonzero(_bits(rh) != _bits(dh)))
            layers.append({"layer": b, "hidden_bf16_mismatch": mm, "exact": mm == 0})
            if mm and first_div is None:
                first_div = b
        result["layers"] = layers
        result["first_divergent_layer"] = first_div
        result["all_layers_exact"] = first_div is None
    result["verdict"] = "PASS" if result["logits_exact"] else "FAIL"
    return result


def verify_generation(model, prompt, max_new_tokens: int = 8, tokenizer=None) -> dict:
    """Greedy decode on both backends; the token sequences must match exactly."""
    if not getattr(model, "is_fully_compiled", True):
        raise RuntimeError("model is not fully compiled — the rail path needs every layer")
    from railnet.runtime.generation import generate

    rail = generate(
        model, prompt, max_new_tokens=max_new_tokens, tokenizer=tokenizer, backend="rail"
    )
    dense = generate(
        model, prompt, max_new_tokens=max_new_tokens, tokenizer=tokenizer, backend="dense"
    )
    exact = rail["tokens"] == dense["tokens"]
    return {
        "prompt": prompt,
        "rail_tokens": rail["tokens"],
        "dense_tokens": dense["tokens"],
        "rail_text": rail.get("text"),
        "token_sequence_exact": exact,
        "verdict": "PASS" if exact else "FAIL",
    }
