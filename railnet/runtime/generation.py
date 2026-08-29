"""Deterministic greedy generation (temperature 0) for a RailNetModel."""

from __future__ import annotations

import contextlib

import numpy as np

from railnet.transformer import block_forward, rms_norm


def _encode(model, prompt, tokenizer):
    if isinstance(prompt, str):
        tok = tokenizer or model.get_tokenizer()
        return list(tok.encode(prompt).ids), tok
    return [int(t) for t in prompt], tokenizer


def _forward_step(model, h, caches, pos_offset):
    ctx = model.ctx
    for b in range(model.n_layers):
        h, caches[b] = block_forward(
            h,
            model._norms[b],
            model._layer_backend(b),
            ctx,
            cache=caches[b],
            pos_offset=pos_offset,
        )
    return model._emb.logits_chunked(rms_norm(h[-1:], model._final_norm, ctx)[0])


def generate(model, prompt, max_new_tokens: int = 32, tokenizer=None) -> dict:
    """Greedy decode. ``prompt`` may be text or a list of token ids.

    Returns ``{prompt, prompt_tokens, tokens, text?}`` (``text`` only when a
    tokenizer is available).
    """
    ids, tok = _encode(model, prompt, tokenizer)
    eos_ids = {int(x) for x in (model.config.get("eos_token_id") or [])}

    caches: list = [None] * model.n_layers
    logits = _forward_step(model, model._emb.rows_f64(ids), caches, pos_offset=0)
    next_tok = int(np.argmax(logits))

    out_tokens = [next_tok]
    pos = len(ids)
    while len(out_tokens) < max_new_tokens and next_tok not in eos_ids:
        logits = _forward_step(model, model._emb.rows_f64([next_tok]), caches, pos_offset=pos)
        next_tok = int(np.argmax(logits))
        out_tokens.append(next_tok)
        pos += 1

    result = {"prompt": prompt, "prompt_tokens": ids, "tokens": out_tokens}
    if tok is not None:
        with contextlib.suppress(Exception):
            result["text"] = tok.decode(out_tokens)
    return result
