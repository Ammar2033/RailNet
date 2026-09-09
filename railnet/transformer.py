"""Transformer operations shared by the dense reference and the RailNet path.

Supports multiple model families:
  * Gemma3: sandwich norms, (1+w) centered RMSNorm, gelu_tanh, sliding window
  * Llama-3 / 3.2: standard RMSNorm (w), SwiGLU (silu(gate) * up), Llama-3 RoPE scaling
  * Qwen-2.5: standard RMSNorm (w), SwiGLU, QKV attention bias support

Only the linear backend differs between dense reference and RailNet; all non-linear
operations (RMSNorm, RoPE, attention, activations) are shared bit-exact.
"""

from __future__ import annotations

import numpy as np

from railnet.dtypes.bf16 import bf16_array_to_float32, fp32_array_to_bf16_bits


# ── KV Cache Buffer ─────────────────────────────────────────────────────────


class KVCache:
    """Preallocated O(1) slice-updating KV cache buffer.

    Avoids repeated memory reallocations (np.concatenate) across autoregressive steps.
    """

    def __init__(
        self,
        kv_heads: int,
        head_dim: int,
        initial_capacity: int = 256,
        dtype=np.float64,
    ):
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.capacity = max(initial_capacity, 16)
        self.len = 0
        self.dtype = dtype
        self.k = np.empty((kv_heads, self.capacity, head_dim), dtype=dtype)
        self.v = np.empty((kv_heads, self.capacity, head_dim), dtype=dtype)

    def update(self, new_k: np.ndarray, new_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert new_k, new_v of shape (kv_heads, seq, head_dim) and return active views."""
        seq = new_k.shape[1]
        req = self.len + seq
        if req > self.capacity:
            new_cap = max(self.capacity * 2, req + 256)
            new_k_buf = np.empty((self.kv_heads, new_cap, self.head_dim), dtype=self.dtype)
            new_v_buf = np.empty((self.kv_heads, new_cap, self.head_dim), dtype=self.dtype)
            if self.len > 0:
                new_k_buf[:, : self.len, :] = self.k[:, : self.len, :]
                new_v_buf[:, : self.len, :] = self.v[:, : self.len, :]
            self.k = new_k_buf
            self.v = new_v_buf
            self.capacity = new_cap

        self.k[:, self.len : req, :] = new_k
        self.v[:, self.len : req, :] = new_v
        self.len = req
        return self.k[:, :req, :], self.v[:, :req, :]


def _ensure_kv_cache(cache, kh: np.ndarray, vh: np.ndarray) -> tuple[np.ndarray, np.ndarray, object]:
    """Helper to support either a KVCache instance, a dict {'K': ..., 'V': ...}, or None."""
    if isinstance(cache, KVCache):
        k_active, v_active = cache.update(kh, vh)
        return k_active, v_active, cache

    if isinstance(cache, dict):
        kh_full = np.concatenate([cache["K"], kh], axis=1)
        vh_full = np.concatenate([cache["V"], vh], axis=1)
        return kh_full, vh_full, {"K": kh_full, "V": vh_full}

    # Initial token prefill when cache is requested
    c = KVCache(kh.shape[0], kh.shape[2], initial_capacity=max(256, kh.shape[1] * 2))
    k_active, v_active = c.update(kh, vh)
    return k_active, v_active, c


# ── Mathematical Primitives ──────────────────────────────────────────────────


def rms_norm(x: np.ndarray, w: np.ndarray, ctx: TransformerContext) -> np.ndarray:
    """RMSNorm with support for standard (w) or centered (1+w) scaling."""
    var = np.mean(x * x, axis=-1, keepdims=True)
    scale = (1.0 + w) if ctx.norm_offset != 0.0 else w
    return x * (1.0 / np.sqrt(var + ctx.eps)) * scale


def rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax_last(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softcap(x: np.ndarray, cap: float | None) -> np.ndarray:
    """Gemma2/3 logit softcapping: cap * tanh(x / cap) (no-op when cap is falsy)."""
    if not cap:
        return x
    return cap * np.tanh(x / cap)


def causal_mask(scores, seq: int, kv_len: int, pos_offset: int, sliding_window: int | None = None):
    qi = pos_offset + np.arange(seq)[:, None]
    kj = np.arange(kv_len)[None, :]
    blocked = kj > qi
    if sliding_window:
        blocked |= kj <= qi - sliding_window
    return np.where(blocked, -np.inf, scores)


def rope_cos_sin(
    positions,
    ctx: TransformerContext,
    base: float | None = None,
    rope_scaling: dict | None = None,
):
    """Compute cos and sin rotary embeddings with optional Meta Llama 3 frequency scaling."""
    if base is None:
        base = ctx.rope_base(0)

    half = np.arange(0, ctx.head_dim, 2, dtype=np.float64)
    inv_freq = base ** -(half / ctx.head_dim)

    # Llama 3 RoPE frequency scaling if configured
    scaling = rope_scaling or getattr(ctx, "rope_scaling", None)
    if scaling and scaling.get("rope_type") == "llama3":
        factor = float(scaling.get("factor", 32.0))
        low_freq_factor = float(scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(scaling.get("high_freq_factor", 4.0))
        old_context_len = float(scaling.get("original_max_position_embeddings", 8192))

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        # wavelen = 2 * pi / freq
        wavelen = 2.0 * np.pi / inv_freq
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth = np.clip(smooth, 0.0, 1.0)
        inv_freq = (1.0 - smooth) * (inv_freq / factor) + smooth * inv_freq

    pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    freqs = pos[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


# ── Transformer Context Base & Implementations ───────────────────────────────


class TransformerContext:
    """Base context specifying model architecture constants and forward wiring."""

    norm_offset: float = 0.0
    layer_norm_keys: tuple[str, ...] = ("input_layernorm", "post_attention_layernorm")
    bias_keys: tuple[str, ...] = ()
    embed_scale: float = 1.0

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg.get("num_key_value_heads", self.heads)
        self.head_dim = cfg.get("head_dim", self.hidden // self.heads)
        self.kv_groups = self.heads // self.kv_heads
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.q_scale = float(cfg.get("query_pre_attn_scalar", self.head_dim) ** -0.5)
        self.attn_softcap = cfg.get("attn_logit_softcapping")
        self.final_softcap = cfg.get("final_logit_softcapping")

    def rope_base(self, layer_idx: int) -> float:
        return float(self.cfg.get("rope_theta", 10000.0))

    def forward_block(
        self,
        h: np.ndarray,
        norms: dict,
        lin,
        cache=None,
        pos_offset: int = 0,
        layer_idx: int = 0,
        biases: dict | None = None,
    ) -> tuple[np.ndarray, object]:
        raise NotImplementedError


class GemmaContext(TransformerContext):
    """Gemma3 transformer architecture specification."""

    norm_offset = 1.0
    layer_norm_keys = (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "q_norm",
        "k_norm",
    )

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.rope_local_base = float(cfg.get("rope_local_base_freq", 10000.0))
        self.rope_global_base = float(cfg.get("rope_theta", self.rope_local_base))
        self.sliding_window = cfg.get("sliding_window")
        self.sliding_window_pattern = int(cfg.get("sliding_window_pattern", 0) or 0)
        # Gemma3 casts the normalizer sqrt(hidden) to BF16
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

    def forward_block(
        self,
        h: np.ndarray,
        norms: dict,
        lin,
        cache=None,
        pos_offset: int = 0,
        layer_idx: int = 0,
        biases: dict | None = None,
    ) -> tuple[np.ndarray, object]:
        seq = h.shape[0]
        residual = h

        hn = rms_norm(h, norms["input_layernorm"], self)
        q = lin("q_proj", hn)
        k = lin("k_proj", hn)
        v = lin("v_proj", hn)

        qh = q.reshape(seq, self.heads, self.head_dim).transpose(1, 0, 2)
        kh = k.reshape(seq, self.kv_heads, self.head_dim).transpose(1, 0, 2)
        vh = v.reshape(seq, self.kv_heads, self.head_dim).transpose(1, 0, 2)

        qh = rms_norm(qh, norms["q_norm"], self)
        kh = rms_norm(kh, norms["k_norm"], self)

        positions = pos_offset + np.arange(seq)
        cos, sin = rope_cos_sin(positions, self, base=self.rope_base(layer_idx))

        if cache is not None:
            kh, vh, cache = _ensure_kv_cache(cache, kh, vh)
        kv_len = kh.shape[1]

        kh_rep = np.repeat(kh, self.kv_groups, axis=0)
        vh_rep = np.repeat(vh, self.kv_groups, axis=0)

        qh = qh * cos[None] + rotate_half(qh) * sin[None]
        kh_rep_rot = kh_rep * cos[None] + rotate_half(kh_rep) * sin[None]

        scores = np.matmul(qh, kh_rep_rot.transpose(0, 2, 1)) * self.q_scale
        scores = softcap(scores, self.attn_softcap)

        window = None if self.is_global_layer(layer_idx) else self.sliding_window
        scores = causal_mask(scores, seq, kv_len, pos_offset, sliding_window=window)

        probs = softmax_last(scores)
        ctx_out = np.matmul(probs, vh_rep)

        attn_out = ctx_out.transpose(1, 0, 2).reshape(seq, self.heads * self.head_dim)
        o = lin("o_proj", attn_out)

        # Gemma sandwich norm
        o = rms_norm(o, norms["post_attention_layernorm"], self)
        h = residual + o
        residual = h

        hff = rms_norm(h, norms["pre_feedforward_layernorm"], self)
        g = lin("gate_proj", hff)
        u = lin("up_proj", hff)
        prod = gelu_tanh(g) * u
        d = lin("down_proj", prod)

        h = rms_norm(d, norms["post_feedforward_layernorm"], self)
        h = residual + h

        return h, cache


class LlamaContext(TransformerContext):
    """Llama-3 / Llama-3.2 transformer architecture specification."""

    norm_offset = 0.0
    layer_norm_keys = ("input_layernorm", "post_attention_layernorm")
    embed_scale = 1.0

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.rope_theta = float(cfg.get("rope_theta", 500000.0))
        self.rope_scaling = cfg.get("rope_scaling")

    def rope_base(self, layer_idx: int) -> float:
        return self.rope_theta

    def forward_block(
        self,
        h: np.ndarray,
        norms: dict,
        lin,
        cache=None,
        pos_offset: int = 0,
        layer_idx: int = 0,
        biases: dict | None = None,
    ) -> tuple[np.ndarray, object]:
        seq = h.shape[0]
        residual = h

        # 1. Pre-attention norm
        hn = rms_norm(h, norms["input_layernorm"], self)

        # 2. QKV projection
        q = lin("q_proj", hn)
        k = lin("k_proj", hn)
        v = lin("v_proj", hn)

        qh = q.reshape(seq, self.heads, self.head_dim).transpose(1, 0, 2)
        kh = k.reshape(seq, self.kv_heads, self.head_dim).transpose(1, 0, 2)
        vh = v.reshape(seq, self.kv_heads, self.head_dim).transpose(1, 0, 2)

        positions = pos_offset + np.arange(seq)
        cos, sin = rope_cos_sin(positions, self, base=self.rope_base(layer_idx), rope_scaling=self.rope_scaling)

        if cache is not None:
            kh, vh, cache = _ensure_kv_cache(cache, kh, vh)
        kv_len = kh.shape[1]

        kh_rep = np.repeat(kh, self.kv_groups, axis=0)
        vh_rep = np.repeat(vh, self.kv_groups, axis=0)

        qh = qh * cos[None] + rotate_half(qh) * sin[None]
        kh_rep_rot = kh_rep * cos[None] + rotate_half(kh_rep) * sin[None]

        scores = np.matmul(qh, kh_rep_rot.transpose(0, 2, 1)) * self.q_scale
        scores = causal_mask(scores, seq, kv_len, pos_offset)

        probs = softmax_last(scores)
        ctx_out = np.matmul(probs, vh_rep)

        attn_out = ctx_out.transpose(1, 0, 2).reshape(seq, self.heads * self.head_dim)
        o = lin("o_proj", attn_out)

        # Standard pre-norm residual connection
        h = residual + o
        residual = h

        # 3. MLP with SwiGLU: silu(gate) * up -> down
        hff = rms_norm(h, norms["post_attention_layernorm"], self)
        g = lin("gate_proj", hff)
        u = lin("up_proj", hff)
        prod = silu(g) * u
        d = lin("down_proj", prod)

        h = residual + d
        return h, cache


class QwenContext(TransformerContext):
    """Qwen-2 / Qwen-2.5 transformer architecture specification."""

    norm_offset = 0.0
    layer_norm_keys = ("input_layernorm", "post_attention_layernorm")
    bias_keys = ("q_proj", "k_proj", "v_proj")
    embed_scale = 1.0

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.rope_theta = float(cfg.get("rope_theta", 1000000.0))

    def rope_base(self, layer_idx: int) -> float:
        return self.rope_theta

    def forward_block(
        self,
        h: np.ndarray,
        norms: dict,
        lin,
        cache=None,
        pos_offset: int = 0,
        layer_idx: int = 0,
        biases: dict | None = None,
    ) -> tuple[np.ndarray, object]:
        seq = h.shape[0]
        residual = h

        # 1. Pre-attention norm
        hn = rms_norm(h, norms["input_layernorm"], self)

        # 2. QKV projection with optional bias
        q = lin("q_proj", hn)
        k = lin("k_proj", hn)
        v = lin("v_proj", hn)

        if biases:
            if "q_proj" in biases:
                q = q + biases["q_proj"]
            if "k_proj" in biases:
                k = k + biases["k_proj"]
            if "v_proj" in biases:
                v = v + biases["v_proj"]

        qh = q.reshape(seq, self.heads, self.head_dim).transpose(1, 0, 2)
        kh = k.reshape(seq, self.kv_heads, self.head_dim).transpose(1, 0, 2)
        vh = v.reshape(seq, self.kv_heads, self.head_dim).transpose(1, 0, 2)

        positions = pos_offset + np.arange(seq)
        cos, sin = rope_cos_sin(positions, self, base=self.rope_base(layer_idx))

        if cache is not None:
            kh, vh, cache = _ensure_kv_cache(cache, kh, vh)
        kv_len = kh.shape[1]

        kh_rep = np.repeat(kh, self.kv_groups, axis=0)
        vh_rep = np.repeat(vh, self.kv_groups, axis=0)

        qh = qh * cos[None] + rotate_half(qh) * sin[None]
        kh_rep_rot = kh_rep * cos[None] + rotate_half(kh_rep) * sin[None]

        scores = np.matmul(qh, kh_rep_rot.transpose(0, 2, 1)) * self.q_scale
        scores = causal_mask(scores, seq, kv_len, pos_offset)

        probs = softmax_last(scores)
        ctx_out = np.matmul(probs, vh_rep)

        attn_out = ctx_out.transpose(1, 0, 2).reshape(seq, self.heads * self.head_dim)
        o = lin("o_proj", attn_out)

        # Standard residual connection
        h = residual + o
        residual = h

        # 3. MLP with SwiGLU
        hff = rms_norm(h, norms["post_attention_layernorm"], self)
        g = lin("gate_proj", hff)
        u = lin("up_proj", hff)
        prod = silu(g) * u
        d = lin("down_proj", prod)

        h = residual + d
        return h, cache


# ── Context Factory ─────────────────────────────────────────────────────────


def create_context(config: dict) -> TransformerContext:
    """Factory creating the appropriate TransformerContext based on model config."""
    model_type = str(config.get("model_type", "")).lower()

    if "gemma" in model_type:
        return GemmaContext(config)
    if "llama" in model_type:
        return LlamaContext(config)
    if "qwen" in model_type:
        return QwenContext(config)

    # Heuristic architecture detection
    if "rope_local_base_freq" in config or "query_pre_attn_scalar" in config:
        return GemmaContext(config)
    if "rope_scaling" in config or float(config.get("rope_theta", 0)) == 500000.0:
        return LlamaContext(config)
    if float(config.get("rope_theta", 0)) == 1000000.0:
        return QwenContext(config)

    # Default to Llama-style decoder
    return LlamaContext(config)


def block_forward(
    h,
    norms,
    lin,
    ctx: TransformerContext,
    cache=None,
    pos_offset=0,
    layer_idx=0,
    biases=None,
):
    """Polymorphic layer forward function forwarding to ctx.forward_block."""
    return ctx.forward_block(
        h,
        norms,
        lin,
        cache=cache,
        pos_offset=pos_offset,
        layer_idx=layer_idx,
        biases=biases,
    )
