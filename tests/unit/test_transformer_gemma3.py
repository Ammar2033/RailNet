"""Gemma3-specific transformer behaviour: local/global RoPE, sliding window,
embedding normalizer."""

import numpy as np

from railnet.transformer import GemmaContext, causal_mask, rope_cos_sin

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
