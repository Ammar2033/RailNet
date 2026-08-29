"""Attention/MLP linear-name bindings (structure per spec)."""

ATTENTION_LINEAR = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

MLP_LINEAR = [
    "gate_proj",
    "up_proj",
    "down_proj",
]

LAYER_LINEAR = ATTENTION_LINEAR + MLP_LINEAR

NORM_KEYS = [
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "q_norm",
    "k_norm",
]
