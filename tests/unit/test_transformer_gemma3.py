"""Gemma3-specific transformer behaviour: local/global RoPE, sliding window,
embedding normalizer."""

import numpy as np

from railnet.transformer import (
    GemmaContext,
    block_forward,
    causal_mask,
    gelu_tanh,
    rms_norm,
    rope_cos_sin,
    rotate_half,
    softmax_last,
)

FULL_CFG = {
    "hidden_size": 1152,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 256,
    "rope_local_base_freq": 10000,
    "rope_theta": 1000000,
    "sliding_window": 512,
    "sliding_window_pattern": 6,
}
MINI_CFG = {
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 4,
    "rope_local_base_freq": 10000,
}


class TestLayerType:
    def test_gemma3_global_every_sixth_layer(self):
        ctx = GemmaContext(FULL_CFG)
        globals_ = [b for b in range(26) if ctx.is_global_layer(b)]
        assert globals_ == [5, 11, 17, 23]

    def test_rope_base_switches(self):
        ctx = GemmaContext(FULL_CFG)
        assert ctx.rope_base(0) == 10000
        assert ctx.rope_base(5) == 1000000

    def test_no_pattern_means_all_global_local_base(self):
        ctx = GemmaContext(MINI_CFG)
        assert ctx.is_global_layer(0) and ctx.is_global_layer(3)
        assert ctx.rope_base(3) == 10000  # rope_theta absent -> local base


class TestRoPEBase:
    def test_base_changes_frequencies(self):
        ctx = GemmaContext(FULL_CFG)
        pos = np.arange(4)
        c_local, _ = rope_cos_sin(pos, ctx, base=ctx.rope_local_base)
        c_global, _ = rope_cos_sin(pos, ctx, base=ctx.rope_global_base)
        assert not np.allclose(c_local, c_global)

    def test_default_base_is_local(self):
        ctx = GemmaContext(FULL_CFG)
        pos = np.arange(4)
        a, _ = rope_cos_sin(pos, ctx)
        b, _ = rope_cos_sin(pos, ctx, base=ctx.rope_local_base)
        assert np.array_equal(a, b)


class TestSlidingWindowMask:
    def test_sliding_window_blocks_far_past(self):
        seq, kv = 4, 10
        scores = np.zeros((1, seq, kv))
        m = causal_mask(scores, seq, kv, pos_offset=6, sliding_window=3)
        # query 0 is at absolute position 6; window 3 keeps keys 4,5,6
        row0 = m[0, 0]
        assert np.isneginf(row0[3]) and not np.isneginf(row0[4])
        assert not np.isneginf(row0[6]) and np.isneginf(row0[7])  # future still masked

    def test_no_window_is_plain_causal(self):
        seq, kv = 3, 3
        m = causal_mask(np.zeros((1, seq, kv)), seq, kv, pos_offset=0)
        assert np.isneginf(m[0, 0, 1]) and not np.isneginf(m[0, 2, 0])


class TestEmbedScale:
    def test_gemma3_1b_normalizer_is_bf16_rounded(self):
        ctx = GemmaContext(FULL_CFG)
        assert ctx.embed_scale == 33.75  # BF16(sqrt(1152)), not 33.941

    def test_mini_normalizer(self):
        ctx = GemmaContext(MINI_CFG)
        assert abs(ctx.embed_scale - 8**0.5) < 0.1


def _reference_layer(h, norms, weights, ctx, layer_idx):
    """Independent hand-written Gemma3 decoder layer (no KV cache, single seq).

    Pins the 'sandwich' norm order: each sub-block is normalized BEFORE the
    residual add.
    """
    seq = h.shape[0]

    def lin(name, x):
        return x @ weights[name].T

    # --- attention branch ---
    x = rms_norm(h, norms["input_layernorm"], ctx)
    q = lin("q_proj", x).reshape(seq, ctx.heads, ctx.head_dim).transpose(1, 0, 2)
    k = lin("k_proj", x).reshape(seq, ctx.kv_heads, ctx.head_dim).transpose(1, 0, 2)
    v = lin("v_proj", x).reshape(seq, ctx.kv_heads, ctx.head_dim).transpose(1, 0, 2)
    q = rms_norm(q, norms["q_norm"], ctx)
    k = rms_norm(k, norms["k_norm"], ctx)

    cos, sin = rope_cos_sin(np.arange(seq), ctx, base=ctx.rope_base(layer_idx))
    q = q * cos[None] + rotate_half(q) * sin[None]
    k = k * cos[None] + rotate_half(k) * sin[None]
    k = np.repeat(k, ctx.kv_groups, axis=0)
    v = np.repeat(v, ctx.kv_groups, axis=0)

    scores = (q @ k.transpose(0, 2, 1)) * ctx.q_scale
    qi = np.arange(seq)[:, None]
    kj = np.arange(seq)[None, :]
    blocked = kj > qi
    if not ctx.is_global_layer(layer_idx) and ctx.sliding_window:
        blocked |= kj <= qi - ctx.sliding_window
    scores = np.where(blocked, -np.inf, scores)
    ctx_out = softmax_last(scores) @ v
    o = lin("o_proj", ctx_out.transpose(1, 0, 2).reshape(seq, ctx.heads * ctx.head_dim))

    o = rms_norm(o, norms["post_attention_layernorm"], ctx)
    h = h + o

    # --- feed-forward branch ---
    x = rms_norm(h, norms["pre_feedforward_layernorm"], ctx)
    mlp = lin("down_proj", gelu_tanh(lin("gate_proj", x)) * lin("up_proj", x))
    mlp = rms_norm(mlp, norms["post_feedforward_layernorm"], ctx)
    return h + mlp


class TestDecoderLayerStructure:
    def _fixtures(self, rng, ctx):
        h = ctx.hidden
        i = 16
        norms = {
            k: rng.standard_normal(h) * 0.1
            for k in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            )
        }
        norms["q_norm"] = rng.standard_normal(ctx.head_dim) * 0.1
        norms["k_norm"] = rng.standard_normal(ctx.head_dim) * 0.1
        w = {
            "q_proj": rng.standard_normal((ctx.heads * ctx.head_dim, h)) * 0.05,
            "k_proj": rng.standard_normal((ctx.kv_heads * ctx.head_dim, h)) * 0.05,
            "v_proj": rng.standard_normal((ctx.kv_heads * ctx.head_dim, h)) * 0.05,
            "o_proj": rng.standard_normal((h, ctx.heads * ctx.head_dim)) * 0.05,
            "gate_proj": rng.standard_normal((i, h)) * 0.05,
            "up_proj": rng.standard_normal((i, h)) * 0.05,
            "down_proj": rng.standard_normal((h, i)) * 0.05,
        }
        return norms, w

    def test_block_forward_matches_independent_reference(self):
        rng = np.random.default_rng(7)
        ctx = GemmaContext(MINI_CFG)
        norms, w = self._fixtures(rng, ctx)
        h = rng.standard_normal((5, ctx.hidden))

        got, _ = block_forward(h, norms, lambda n, x: x @ w[n].T, ctx, layer_idx=0)
        expect = _reference_layer(h, norms, w, ctx, 0)
        assert np.allclose(got, expect, atol=1e-12)
