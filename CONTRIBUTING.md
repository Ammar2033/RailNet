# Contributing to RailNet

Thank you for your interest in contributing to RailNet!

RailNet is an experimental research architecture that radically changes how neural network parameters are stored and executed. Because of its emphasis on **exactness** and **memory tracking**, contributions must adhere to strict validation guidelines.

## Development Setup

1. Clone the repository and navigate into it.
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
3. Install development dependencies (if any):
   ```bash
   pip install pytest numpy
   ```

## Architecture and Design Principles

- **No Dense Parameters at Runtime**: RailNet's core promise is that the `(out_features, in_features)` dense floating-point matrices are completely removed from the execution path.
- **Bit-Level Exactness**: Any new compiler algorithms, dtype registrations, or memory mapping structures MUST mathematically reconstruct the exact original bit patterns. We do not do lossy quantization.
- **Honest Memory Reporting**: If your code uses memory, it must be tracked and reported via the `MemoryBudget` subsystem. 

## Testing

All code changes must pass the internal test suite.

```bash
python -m pytest tests/ -v
```

### Exactness Tests

The most critical tests in RailNet are the `test_exact_tensor.py` verifications. If you modify the core routing assignment (`railnet/rails/_compile.py`) or the shared rails generator (`railnet/rails/_learner.py`), you must ensure that Tier 1 (Weight Exactness) and Tier 2 (Mathematical Oracle) tests continue to pass.

## Adding Support for New Model Architectures

If you wish to add support for a new model family (e.g., Llama, Qwen):
1. Create a new subclass of `ModelAdapter` in `railnet/models/`.
2. Implement `build_graph()` to parse the layer structure.
3. Update `railnet/models/registry.py` to register your adapter.
4. Add unit tests for your adapter in `tests/unit/test_models.py`.

## Pull Request Process

1. Ensure all tests pass.
2. Ensure you haven't introduced any arbitrary rounding in the `railnet.dtypes` module.
3. Keep PRs focused on single architectural changes or specific module optimizations.
4. Update `CHANGELOG.md` with your changes.
