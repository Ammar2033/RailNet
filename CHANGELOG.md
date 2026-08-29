# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hardware Abstraction Layer (`railnet.hardware`)**: Implemented software emulations of routing and compute fabrics to simulate physical spatial constraints of a theoretical ASIC/FPGA design.
- **Model Adapter Registry (`railnet.models`)**: Expandable adapter architecture for handling different model families (Gemma, LLaMA, Qwen).
- **Core Abstractions (`railnet.core`)**: Formal `RailTensor` and `RailGraph` classes to act as primary memory and representation data structures.
- **Exactness Tests (`railnet.verification`)**: test suite validating compilation determinism, routing correctness, and bit-level math equality.
- **Bulk model compiler (`railnet.compiler.model.compile_model`)**: compiles every dense linear of a safetensors model into a RailNet artifact directory (`manifest.json` + per-layer rail tables + route-id maps).
- **Working CPU runtime (`railnet.runtime.RailNetModel`)**: `forward()` / `generate()` now execute a real exact forward through the rail kernel — no dense weight array — with norms + tied embedding streamed from the source safetensors. Covered by `tests/exactness/test_end_to_end_runtime.py` (synthetic model, BF16-bitwise logit equality vs a dense reference).
- **Verification hierarchy in code (`railnet.verification`)**: `rail_oracle` / `dense_oracle` (Fraction-exact linear), `verify_forward` (rail path vs dense reference of the same graph, BF16-bitwise, full vocab + per layer), `verify_generation` (greedy tokens), `verify_compiled` (structural check of a compiled directory). `RailNetModel` gained a `dense` linear backend used only as the reference. `railnet verify compiled/` now validates a directory.
- **Reproduction scripts**: `research/reproduce_gemma.py` (compile → verify → generate) and `research/crosscheck_hf.py` (whole graph vs `transformers` Gemma3, `[hf]` extra).
- **Expanded Documentation**: New guides detailing exactly how memory compression (`MEMORY.md`), hardware mapping (`HARDWARE.md`), and mathematical exactness (`EXACTNESS.md`) behave.

### Changed
- **Compiler Rewrite (`railnet.compiler`)**: Re-architected compiler to produce type-safe `RailTensor` artifacts rather than raw unstructured dictionaries.
- **Dtype Management (`railnet.dtypes`)**: Moved hardcoded type comparisons into a robust registry-based generic DType wrapper.
- **Project Structure (`railnet.rails`)**: The monolithic prototype compiler was extracted into formal submodules (`_analysis.py`, `_compile.py`, `_repair.py`, `_optimize.py`), reformatted with `ruff format`, and the tree is now clean under `ruff check` and `mypy`.
- **Runtime Modularity (`railnet.runtime`)**: Removed the remaining `importlib.util` dynamic-import shims (`runtime/linear.py`, `runtime/generation.py`, `compiler/model.py`); modules now use plain imports and explicit `GemmaContext`.

### Fixed
- **Gemma3 decoder-layer norm order**: `block_forward` added the attention residual *before*
  `post_attention_layernorm`; Gemma normalizes the attention branch first, then adds the
  residual (the "sandwich" norm). Rail ≡ dense held throughout since both paths shared the
  bug — it only surfaced against a reference implementation.
- `block_forward` now models per-layer local/global RoPE base, the sliding-window mask, the
  BF16 `sqrt(hidden)` embedding normalizer, and optional Gemma2 logit softcapping.
- `railnet verify <manifest.json>` crashed with `ModuleNotFoundError: railnet.artifact`; added `railnet.artifacts.verify_checksum`.
- `railnet.validation` and `railnet.runtime.attention` raised `ImportError` on import (stale module paths).
- `verify_block` was calling `block_forward` without the required `ctx`; `verification.oracle.rail_oracle` was a non-functional stub.
- `NameError: REPAIR_COMPILE_BUDGET` in `railnet.rails._repair`; missing `numpy` import in `railnet.rails.learner`.
- `EmbeddingMMap` leaked the backing file descriptor on `close()`.

### Removed
- Legacy root-level monolith modules superseded by the internal `railnet/` package.
- Duplicate root-level copies of the `research/experiments/` scripts, `main.py`, and loose scratch/log files.

## [0.1.0] - 2026-08-25
### Added
- Initial Stage 15b Gemma3 1B Proof of Concept.
- Lossless BF16 compilation proving a 96-rail fabric is capable of representing the continuous LLM parameter space.
- Experimental Oracle verification using integer fractions.
