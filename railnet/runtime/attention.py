"""Attention runtime — re-exports the shared transformer ops.

Attention itself lives inside :func:`railnet.transformer.block_forward`;
there is no standalone ``attention_block``.
"""

from railnet.transformer import block_forward, rms_norm, rope_cos_sin, softmax_last

__all__ = ["block_forward", "rms_norm", "rope_cos_sin", "softmax_last"]
