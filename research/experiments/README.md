# research/experiments/

Preserved research history — the 01–16 scripts that established the RailNet method
(exact topology, shared multiplication, Gemma BF16 analysis, block/multi-block execution,
KV cache, full forward, generation).

**These target a pre-refactor package API** (`transformer.init_from_config`, `ctx`-less
`block_forward`, the old `railnet.artifact` module) and do not run as-is against the current
`railnet/` package. They are kept for provenance, not as runnable entry points.

Current equivalents:

| Old script | Now |
|---|---|
| `15a_gemma_full_compile.py` | `railnet.compiler.model.compile_model` / `railnet compile` |
| `15b_gemma_full_forward.py` | `railnet.verification.verify_forward` / `research/reproduce_gemma.py` |
| `16_gemma_generation.py` | `railnet.runtime.RailNetModel.generate` / `railnet.verification.verify_generation` |
| `05_bf16_exact_kernel_oracle.py` | `railnet.verification.rail_oracle` / `dense_oracle` |
