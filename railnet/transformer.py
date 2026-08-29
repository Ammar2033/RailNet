"""Gemma3 transformer ops shared by the dense reference and the RailNet path.

Only the linear backend differs between the two paths; every non-linear op
(RMSNorm, RoPE, attention, GELU) is the SAME function for both. A BF16-bitwise
match between the paths therefore isolates the rail kernel + routing.

Gemma3 specifics modelled here (activated by the corresponding config keys):
  * embedding scaled by BF16(sqrt(hidden_size))
  * per-layer local vs global RoPE base (``rope_local_base_freq`` /
    ``rope_theta``, global every ``sliding_window_pattern``-th layer)
  * sliding-window causal mask on local layers (``sliding_window``)

Full token-for-token equivalence with HuggingFace ``modeling_gemma3`` is a
separate validation item — see docs/EXACTNESS.md.
"""

import numpy as np

from railnet.dtypes.bf16 import bf16_array_to_float32, fp32_array_to_bf16_bits


class GemmaContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.kv_groups = self.heads // self.kv_heads
        self.eps = cfg["rms_norm_eps"]
        self.q_scale = cfg["query_pre_attn_scalar"] ** -0.5
        self.rope_local_base = cfg["rope_local_base_freq"]
        self.rope_global_base = cfg.get("rope_theta", self.rope_local_base)
        self.sliding_window = cfg.get("sliding_window")
        self.sliding_window_pattern = int(cfg.get("sliding_window_pattern", 0) or 0)
        # Gemma2-style logit softcapping (null on Gemma3 1B, honoured if present).
        self.attn_softcap = cfg.get("attn_logit_softcapping")
        self.final_softcap = cfg.get("final_logit_softcapping")
        # Gemma3 casts the normalizer to the activation dtype (BF16) before use.
        self.embed_scale = float(
            bf16_array_to_float32(
                fp32_array_to_bf16_bits(np.array([self.hidden**0.5], dtype=np.float32))
            )[0]
        )

    def is_global_layer(self, layer_idx: int) -> bool:
        if not self.sliding_window_pattern:
            return True
        return (layer_idx + 1) % self.sliding_window_pattern == 0

    def rope_base(self, layer_idx: int) -> float:
        return self.rope_global_base if self.is_global_layer(layer_idx) else self.rope_local_base


def rms_norm(x, w, ctx: GemmaContext):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + ctx.eps)) * (1.0 + w)


def rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(positions, ctx: GemmaContext, base: float | None = None):
    if base is None:
        base = ctx.rope_local_base
    half = np.arange(0, ctx.head_dim, 2, dtype=np.float64)
    inv_freq = base ** -(half / ctx.head_dim)
    pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    freqs = pos[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def gelu_tanh(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def softmax_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softcap(x, cap):
    """Gemma2 logit softcapping: ``cap * tanh(x / cap)`` (no-op when cap is falsy)."""
    if not cap:
        return x
    return cap * np.tanh(x / cap)


def causal_mask(scores, seq, kv_len, pos_offset, sliding_window=None):
    qi = pos_offset + np.arange(seq)[:, None]
    kj = np.arange(kv_len)[None, :]
    blocked = kj > qi
    if sliding_window:
        blocked |= kj <= qi - sliding_window
    return np.where(blocked, -np.inf, scores)


def block_forward(
    h,
    norms,  # dict of the 6 norm vectors for a layer
    lin,  # callable(short_name, x2d) -> y2d
    ctx: GemmaContext,
    cache=None,  # per-block KV cache dict or None
    pos_offset=0,
    layer_idx=0,
):
    """One Gemma3 decoder layer with sandwich norms and an optional growing KV
    cache. Returns ``(out, cache)``."""
    seq = h.shape[0]
    residual = h

    hn = rms_norm(h, norms["input_layernorm"], ctx)
    q = lin("q_proj", hn)
    k = lin("k_proj", hn)
    v = lin("v_proj", hn)

    qh = q.reshape(seq, ctx.heads, ctx.head_dim).transpose(1, 0, 2)
    kh = k.reshape(seq, ctx.kv_heads, ctx.head_dim).transpose(1, 0, 2)
    vh = v.reshape(seq, ctx.kv_heads, ctx.head_dim).transpose(1, 0, 2)

    qh = rms_norm(qh, norms["q_norm"], ctx)
    kh = rms_norm(kh, norms["k_norm"], ctx)

    positions = pos_offset + np.arange(seq)
    cos, sin = rope_cos_sin(positions, ctx, base=ctx.rope_base(layer_idx))

    if cache is not None:
        kh = np.concatenate([cache["K"], kh], axis=1)
        vh = np.concatenate([cache["V"], vh], axis=1)

    cache = {"K": kh, "V": vh}
    kv_len = kh.shape[1]

    kh_rep = np.repeat(kh, ctx.kv_groups, axis=0)
    vh_rep = np.repeat(vh, ctx.kv_groups, axis=0)

    qh = qh * cos[None] + rotate_half(qh) * sin[None]
    kh_rep_rot = kh_rep * cos[None] + rotate_half(kh_rep) * sin[None]

    scores = np.matmul(qh, kh_rep_rot.transpose(0, 2, 1)) * ctx.q_scale
    scores = softcap(scores, ctx.attn_softcap)

    window = None if ctx.is_global_layer(layer_idx) else ctx.sliding_window
    scores = causal_mask(scores, seq, kv_len, pos_offset, sliding_window=window)

    probs = softmax_last(scores)
    ctx_out = np.matmul(probs, vh_rep)

    attn_out = ctx_out.transpose(1, 0, 2).reshape(seq, ctx.heads * ctx.head_dim)
    o = lin("o_proj", attn_out)

    # Gemma "sandwich" norm: normalize the attention branch, THEN add residual.
    o = rms_norm(o, norms["post_attention_layernorm"], ctx)
    h = residual + o
    residual = h

    hff = rms_norm(h, norms["pre_feedforward_layernorm"], ctx)
    g = lin("gate_proj", hff)
    u = lin("up_proj", hff)
    prod = gelu_tanh(g) * u
    d = lin("down_proj", prod)

    h = rms_norm(d, norms["post_feedforward_layernorm"], ctx)
    h = residual + h

    return h, cache
