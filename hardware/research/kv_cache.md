# KV cache — research

KV cache is **dynamic runtime state**, not part of the RailNet artifact
(spec 3, §28). RailNet's rail representation does not touch it — K and V are
plain activations. So the KV cache design is the same problem any transformer
accelerator has, and it is orthogonal to the rail fabric.

## Size (Gemma3 1B, per token, both K and V, BF16)

`2 · num_layers · num_kv_heads · head_dim · 2 bytes`
= `2 · 26 · 1 · 256 · 2` = **26 KiB / token**. GQA (1 KV head) keeps this small.
32k context ≈ 832 MiB.

Larger targets (§28) scale with `num_layers · num_kv_heads · head_dim ·
context · batch · precision` — compute per model class when those numbers are
picked, not before.

## Options

on-chip SRAM / BRAM / URAM for short context; external LPDDR / DDR / HBM for
long context or batch; paged KV; compressed / quantized KV. Multi-chip
distribution is a Phase-16 question.

Status: `RESEARCH`, low priority — decouple from the rail fabric work.
