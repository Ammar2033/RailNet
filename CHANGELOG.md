# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hardware Abstraction Layer (`railnet.hardware`)**: Implemented software emulations of routing and compute fabrics to simulate physical spatial constraints of a theoretical ASIC/FPGA design.
- **Model Adapter Registry (`railnet.models`)**: Expandable adapter architecture for handling different model families (Gemma, LLaMA, Qwen).
- **Core Abstractions (`railnet.core`)**: Formal `RailTensor` and `RailGraph` classes to act as primary memory and representation data structures.
- **Exactness Tests (`railnet.verification`)**: 196 test cases validating compilation determinism, routing correctness, and bit-level fraction math equality.
- **Expanded Documentation**: New guides detailing exactly how memory compression (`MEMORY.md`), hardware mapping (`HARDWARE.md`), and mathematical exactness (`EXACTNESS.md`) behave.

### Changed
- **Compiler Rewrite (`railnet.compiler`)**: Re-architected compiler to produce type-safe `RailTensor` artifacts rather than raw unstructured dictionaries.
- **Dtype Management (`railnet.dtypes`)**: Moved hardcoded type comparisons into a robust registry-based generic DType wrapper.
- **Project Structure (`railnet.rails`)**: The monolithic 3.5k-line prototype compiler (`04_bf16_learned_basis.py`) was fully extracted, chunked, and moved into formal submodule structures (`_analysis.py`, `_compile.py`, `_repair.py`, `_optimize.py`).
- **Runtime Modularity (`railnet.runtime`)**: Eradicated all `importlib.util` dynamic imports that were polluting package paths. Runtime modules now cleanly leverage dependency injection and explicit contexts (e.g., `GemmaContext`).

### Removed
- Legacy root-level monolith modules (`compiler.py`, `bf16.py`, `basis.py`, `topology.py`, `artifact.py`, `mlp.py`) which were superseded by the internal `railnet/` package.
- All loose root experimental logs (`.log`, `.json`, `.csv`) which are now cleanly organized into `results/data/`.

## [0.1.0] - 2026-08-25
### Added
- Initial Stage 15b Gemma3 1B Proof of Concept.
- Lossless BF16 compilation proving a 96-rail fabric is capable of representing the continuous LLM parameter space.
- Experimental Oracle verification using integer fractions.
