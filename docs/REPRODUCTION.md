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

## Running the Exactness Tests

To reproduce our claims of absolute bit-level exactness without rounding degradation, run the test suite:

```bash
python -m pytest tests/exactness/test_exact_tensor.py -v
```

This test will:
1. Generate random dense matrices.
2. Compile them into a RailNet representation.
3. Perform sparse accumulation based on the route map.
4. Verify that the bits resulting from the accumulation identically match the bits of the original dense matrix.

## Running the Experiments

To run the formal research experiments mapping a whole Gemma block, use the provided scripts in `research/experiments/`:
```bash
python research/experiments/12_gemma_linear_runner.py
```
This script acts as the final stage runner, evaluating weight exactness, math exactness (Fraction Oracle), and output exactness.
