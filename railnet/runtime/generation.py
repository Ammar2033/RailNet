"""Deterministic greedy generation (temperature 0) for a RailNetModel."""

from __future__ import annotations

import contextlib

import numpy as np

from railnet.transformer import rms_norm


def _encode(model, prompt, tokenizer):
    if isinstance(prompt, str):
        tok = tokenizer or model.get_tokenizer()
        return list(tok.encode(prompt).ids), tok
    return [int(t) for t in prompt], tokenizer


def _step_logits(model, h, caches, pos_offset, backend):
    h, _ = model.run_layers(h, caches, pos_offset, backend=backend)
    return model._emb.logits_chunked(rms_norm(h[-1:], model._final_norm, model.ctx)[0])


def generate(
    model, prompt, max_new_tokens: int = 32, tokenizer=None, backend: str = "rail"
) -> dict:
    """Greedy decode. ``prompt`` may be text or a list of token ids.

    Returns ``{prompt, prompt_tokens, tokens, text?}`` (``text`` only when a
    tokenizer is available).
    """
    ids, tok = _encode(model, prompt, tokenizer)
    eos_ids = {int(x) for x in (model.config.get("eos_token_id") or [])}

    caches: list = [None] * model.n_layers
    logits = _step_logits(model, model.embed(ids), caches, 0, backend)
    next_tok = int(np.argmax(logits))

    out_tokens = [next_tok]
    pos = len(ids)
    while len(out_tokens) < max_new_tokens and next_tok not in eos_ids:
        logits = _step_logits(model, model.embed([next_tok]), caches, pos, backend)
        next_tok = int(np.argmax(logits))
        out_tokens.append(next_tok)
        pos += 1

    result = {"prompt": prompt, "prompt_tokens": ids, "tokens": out_tokens}
    if tok is not None:
        with contextlib.suppress(Exception):
            result["text"] = tok.decode(out_tokens)
    return result
