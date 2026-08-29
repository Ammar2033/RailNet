# Reproducing RailNet Results

RailNet is built on strict determinism. This guide will walk you through compiling a tensor into the RailNet spatial format and verifying its exactness.

## Prerequisites
- Download a sample model like `gemma3-1b` into `model_data/` (or run a mock test using random data).

## Running the Compiler

The `RailNetCompiler` is responsible for generating the shared rails and the routing table from dense matrices.
You can run the compiler programmatically:

```python
from railnet.compiler import RailNetCompiler
import numpy as np

# A sample dense weight tensor in bf16
raw = np.random.normal(0, 1, (1024, 1024)).astype(np.float32)

compiler = RailNetCompiler(model="gemma3", default_dtype="bf16")

# Compile with 96 shared rails
compiled_tensor = compiler.compile_tensor(raw, dtype="bf16", rails=96)

# Access the generated RailNet structures
rails = compiled_tensor.rails
routes = compiled_tensor.route_ids
```

## Compiling a whole model

```python
from railnet.compiler.model import compile_model

compile_model("model_data/model.safetensors", out_dir="compiled", rails=96, max_terms=4)
```

This writes `compiled/manifest.json` plus per-layer rail tables and route-id
maps. `--only` / `--limit` (CLI) or the same kwargs (Python) restrict the run
to a subset while iterating; `max_iters` caps the basis-learning loop.

## Running the runtime

```python
from railnet.runtime import RailNetModel

model = RailNetModel.load("compiled")
logits = model.forward([2, 133, 40])        # rail path, no dense weight array
ref    = model.forward_dense([2, 133, 40])  # dense reference, same graph
```

## Full Gemma reproduction

```bash
git lfs pull                       # fetch model_data/model.safetensors (~2 GB)
python research/reproduce_gemma.py  # compile -> verify_compiled -> verify_forward -> verify_generation
```

Writes `results/gemma_repro.json`. The `PASS` verdict means the rail path is BF16-bitwise
identical to the dense computation of the same transformer graph (full vocab + per layer +
greedy tokens). Use `--limit N` / `--only SUBSTR` / `--max-iters` to iterate on a subset;
`--skip-compile` reuses an existing `compiled/`.

## Running the exactness tests

```bash
python -m pytest tests/exactness/ -v
```

`test_exact_tensor.py` checks weight-reconstruction exactness on random matrices.
`test_end_to_end_runtime.py` builds a synthetic Gemma-shaped model, compiles every linear,
and runs the full verification hierarchy (Fraction oracle → kernel → forward → generation),
BF16-bitwise, with no model weights required — this is the CI-sized version of
`reproduce_gemma.py`.

## Research history

`research/experiments/` holds the original 01–16 scripts. They target a pre-refactor API and
are kept for provenance only — see `research/experiments/README.md` for the current equivalents.
This script acts as the final stage runner, evaluating weight exactness, math exactness (Fraction Oracle), and output exactness.
