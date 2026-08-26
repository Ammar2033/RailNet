"""Gemma3 transformer ops shared by dense and railnet paths.

All semantics verified against official HF modeling_gemma3.py
and local config.json (Stage 13). Non-linear ops are the SAME
functions for both paths; only the linear backend differs.
"""
import numpy as np

from . import bf16 as B

CFG = None  # set by init_from_config(cfg)


def init_from_config(cfg):

    global CFG, HIDDEN, HEADS, KV_HEADS, HEAD_DIM
    global KV_GROUPS, EPS, Q_SCALE, ROPE_LOCAL_BASE

    CFG = cfg

    HIDDEN = cfg["hidden_size"]

    HEADS = cfg["num_attention_heads"]

    KV_HEADS = cfg["num_key_value_heads"]

    HEAD_DIM = cfg["head_dim"]

    KV_GROUPS = HEADS // KV_HEADS

    EPS = cfg["rms_norm_eps"]

    Q_SCALE = cfg["query_pre_attn_scalar"] ** -0.5

    ROPE_LOCAL_BASE = cfg["rope_local_base_freq"]


def rms_norm(x, w):

    var = np.mean(x * x, axis=-1, keepdims=True)

    return (
        x
        * (1.0 / np.sqrt(var + EPS))
        * (1.0 + w)
    )


def rotate_half(x):

    d = x.shape[-1]

    x1 = x[..., : d // 2]

    x2 = x[..., d // 2:]

    return np.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(positions):

    half = np.arange(0, HEAD_DIM, 2, dtype=np.float64)

    inv_freq = ROPE_LOCAL_BASE ** -(half / HEAD_DIM)

    pos = np.atleast_1d(
        np.asarray(positions, dtype=np.float64)
    )

    freqs = pos[:, None] * inv_freq[None, :]

    emb = np.concatenate([freqs, freqs], axis=-1)

    return np.cos(emb), np.sin(emb)


def gelu_tanh(x):

    c = np.sqrt(2.0 / np.pi)

    return 0.5 * x * (
        1.0 + np.tanh(c * (x + 0.044715 * x ** 3))
    )


def softmax_last(x):

    e = np.exp(x - x.max(axis=-1, keepdims=True))

    return e / e.sum(axis=-1, keepdims=True)


def attention_block(qh, kh, vh, cos, sin):
    """
    qh: (H, seq, hd) already q_norm'ed
    kh/vh: (KV, seq, hd) k_norm'ed
    Applies RoPE, GQA repeat, causal scores, softmax, ctx.
    Returns ctx reshaped to (seq, H*hd) and scores.
    """

    seq = qh.shape[1]

    qh = qh * cos[None] + rotate_half(qh) * sin[None]

    kh = kh * cos[None] + rotate_half(kh) * sin[None]

    kh_rep = np.repeat(kh, KV_GROUPS, axis=0)

    vh_rep = np.repeat(vh, KV_GROUPS, axis=0)

    scores = np.matmul(
        qh, kh_rep.transpose(0, 2, 1)
    ) * Q_SCALE

    mask = np.triu(
        np.ones((seq, seq), dtype=bool), k=1
    )

    scores = np.where(mask[None], -np.inf, scores)

    probs = softmax_last(scores)

    ctx = np.matmul(probs, vh_rep)

    return (
        ctx.transpose(1, 0, 2).reshape(
            seq, HEADS * HEAD_DIM
        ),
        scores,
    )


def block_forward(
    h,
    norms,          # dict of the 6 norm vectors for a layer
    lin,            # callable(short_name, x2d) -> y2d
    cache=None,     # per-block KV cache dict or None
    pos_offset=0,
):
    """
    One Gemma3 decoder layer with sandwich norms and an
    optional growing KV cache. Returns (out, cache).
    """

    seq = h.shape[0]

    residual = h

    hn = rms_norm(h, norms["input_layernorm"])

    q = lin("q_proj", hn)

    k = lin("k_proj", hn)

    v = lin("v_proj", hn)

    qh = q.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)

    kh = k.reshape(seq, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

    vh = v.reshape(seq, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

    qh = rms_norm(qh, norms["q_norm"])

    kh = rms_norm(kh, norms["k_norm"])

    positions = pos_offset + np.arange(seq)

    cos, sin = rope_cos_sin(positions)

    if cache is not None:

        kh = np.concatenate([cache["K"], kh], axis=1)

        vh = np.concatenate([cache["V"], vh], axis=1)

    cache = {"K": kh, "V": vh}

    kv_len = kh.shape[1]

    kh_rep = np.repeat(kh, KV_GROUPS, axis=0)

    vh_rep = np.repeat(vh, KV_GROUPS, axis=0)

    scores = np.matmul(
        qh, kh_rep.transpose(0, 2, 1)
    ) * Q_SCALE

    qi = pos_offset + np.arange(seq)[:, None]

    kj = np.arange(kv_len)[None, :]

    scores = np.where(
        kj > qi,
        -np.inf,
        scores,
    )

    probs = softmax_last(scores)

    ctx = np.matmul(probs, vh_rep)

    attn_out = ctx.transpose(1, 0, 2).reshape(
        seq, HEADS * HEAD_DIM
    )

    o = lin("o_proj", attn_out)

    h = residual + o

    h = rms_norm(h, norms["post_attention_layernorm"])

    residual = h

    hff = rms_norm(h, norms["pre_feedforward_layernorm"])

    g = lin("gate_proj", hff)

    u = lin("up_proj", hff)

    prod = gelu_tanh(g) * u

    d = lin("down_proj", prod)

    h = rms_norm(d, norms["post_feedforward_layernorm"])

    h = residual + h

    return h, cache
