"""Gemma3 transformer ops shared by dense and railnet paths.

All semantics verified against official HF modeling_gemma3.py
and local config.json (Stage 13). Non-linear ops are the SAME
functions for both paths; only the linear backend differs.
"""
import numpy as np
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


def rms_norm(x, w, ctx: GemmaContext):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + ctx.eps)) * (1.0 + w)


def rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(positions, ctx: GemmaContext):
    half = np.arange(0, ctx.head_dim, 2, dtype=np.float64)
    inv_freq = ctx.rope_local_base ** -(half / ctx.head_dim)
    pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    freqs = pos[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def gelu_tanh(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def softmax_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def block_forward(
    h,
    norms,          # dict of the 6 norm vectors for a layer
    lin,            # callable(short_name, x2d) -> y2d
    ctx: GemmaContext,
    cache=None,     # per-block KV cache dict or None
    pos_offset=0,
):
    """
    One Gemma3 decoder layer with sandwich norms and an
    optional growing KV cache. Returns (out, cache).
    """
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
    cos, sin = rope_cos_sin(positions, ctx)

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

    qi = pos_offset + np.arange(seq)[:, None]
    kj = np.arange(kv_len)[None, :]
    scores = np.where(kj > qi, -np.inf, scores)

    probs = softmax_last(scores)
    ctx_out = np.matmul(probs, vh_rep)

    attn_out = ctx_out.transpose(1, 0, 2).reshape(seq, ctx.heads * ctx.head_dim)
    o = lin("o_proj", attn_out)

    h = residual + o
    h = rms_norm(h, norms["post_attention_layernorm"], ctx)
    residual = h

    hff = rms_norm(h, norms["pre_feedforward_layernorm"], ctx)
    g = lin("gate_proj", hff)
    u = lin("up_proj", hff)
    prod = gelu_tanh(g) * u
    d = lin("down_proj", prod)

    h = rms_norm(d, norms["post_feedforward_layernorm"], ctx)
    h = residual + h

    return h, cache
