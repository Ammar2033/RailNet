# RailNet — Lossless Topology-Driven Neural Execution

> **RailNet is an open-source research project for lossless topology-driven neural-network execution and a future PCIe-attached AI accelerator.**

RailNet replaces dense runtime weight arrays with **shared primitive rails + topology + routing** while preserving the exact same mathematical information (BF16-bitwise).

```
Model → RailNet Compiler → Model-specific RailNet Artifact → RailNet Runtime → CPU/GPU simulation → PCIe RailNet Accelerator → FPGA/ASIC
```

## What is RailNet?

Neural-network weights are not stored as dense `W[i,j]` arrays at runtime. Instead:

* **Rails** — shared BF16 primitives `R0 … R95` (e.g. `R0=+0.03149`)
* **Topology** — per-weight route `W = +R3 - R17 + R42` (signed rail indices, no per-weight coefficient)
* **Routing** — bit-pattern indexed: `bf16_bits → route`
* **Shared computation** — `Y[j] = Σ_r R_r · Σ_i sign(i,j,r)·X[i]` collapses many multiplies into one per rail

Same physical fabric can execute different models by reprogramming rail values + routing (model-specific artifact, hardware-rule-independent).

## Verified Results (PROVEN)

Gemma3 1B class (`hidden=1152, layers=26, vocab=262144, BF16`):

* **182 / 182 tensors lossless, 0 failures** (Stage 15A)
* **26 / 26 layers PASS; 1,048,576 / 1,048,576 logits BF16-exact** (Stage 15B)
* Embedding: exact mmap row lookup (NOT compressed)
* Runtime dense linear weight arrays: **ABSENT**
* Shared multiplication reduction: **≥93.31% full model** (≈95.97% on layer-0 global fabric)

## What is NOT claimed

* No dramatic total bit-storage compression yet — route-map cost is ~dense storage (honest ≈1.23× on layer-0). See `docs/MEMORY.md`.
* No ASIC/FPGA throughput speedup claimed until built and measured.
* 32B / 64B / 128B multi-chip capacities are **architectural goals**, not proven.
* FP16/FP32/INT8 compilers are `READY/PLANNED`, only **BF16 PROVEN**.

## Installation

```bash
git clone https://github.com/Ammar2033/RailNet && cd RailNet
pip install -e ".[dev]"          # + [gemma] for the tokenizer-backed demo
```

Reproducing the Gemma path needs `model_data/model.safetensors` + `config.json`
+ `tokenizer.json` (not bundled). The test suite proves the full pipeline on a
synthetic model and needs none of that.

## Quick Demo

```bash
railnet inspect  model_data/model.safetensors --model gemma3
railnet compile  model_data/model.safetensors --dtype bf16 --rails 96 --terms 4 --out compiled
railnet verify   compiled/manifest.json
railnet generate compiled --prompt "Hello" --max-tokens 32     # needs [gemma] + tokenizer.json
```

`compile` is exhaustive and slow on a 1B model — use `--only` / `--limit` to
compile a subset while iterating.

Python:

```python
from railnet.compiler.model import compile_model
from railnet.runtime import RailNetModel

compile_model("model_data/model.safetensors", out_dir="compiled", rails=96, max_terms=4)

model = RailNetModel.load("compiled")
logits = model.forward([2, 133, 40])              # final-token logits, no dense weights
out = model.generate("Merhaba", max_new_tokens=64)
```

Single tensor:

```python
from railnet.compiler import RailNetCompiler

compiler = RailNetCompiler(model="gemma3")
tensor = compiler.compile_tensor(raw_bits, dtype="bf16", rails=96, max_terms=4)
```

## Architecture

See `ARCHITECTURE.md` and `SPEC.md`.

* `railnet/dtypes/` — generic dtype abstraction (`RailDType`)
* `railnet/rails/` — rail + basis + learner
* `railnet/topology/` — routes + scheduler
* `railnet/compiler/` — model-independent compiler + Gemma/Llama/Qwen adapters
* `railnet/runtime/` — CPU simulator, KV cache, generation
* `railnet/artifacts/` — `.rnmodel` container (magic `RNET`, SHA-256, no dense weights)
* `railnet/hardware/` — PCIe card research model
* `research/experiments/` — preserved 01–16 scripts

## Limitations

* Current CPU runtime is correctness-first (NumPy), not optimized.
* Route-map storage remains the main open problem (`railnet/topology_compression/`).
* Embedding compression not claimed.
* Hardware is research / not yet built.
* Multi-chip scaling not yet implemented.

## Roadmap

`ROADMAP.md` — 16 phases from research proof (✅) → optimized runtime → PCIe → FPGA → ASIC → multi-chip.

## Contributing

See `CONTRIBUTING.md`. Every PR needs tests + docs + honest benchmarks.

## Citation

See `CITATION.cff`.

## License

Apache-2.0
"# RailNet" 
"# RailNet" 
