# EXACTNESS IN RAILNET

RailNet achieves **bit-level exactness** for pre-trained neural networks. This is not a "low-loss" or "high-fidelity" quantization. It is fundamentally mathematically exact, reconstructing the original dense bits flawlessly under the RailNet sparse representation.

## Core Properties

1. **Lossless Recovery**: Given a tensor $W$, the RailNet compiler guarantees that the reconstructed tensor $W_{rec}$ will satisfy $W = W_{rec}$ at the exact bit-level for every element (e.g., matching the `uint16` representation of BF16 identically).
2. **Deterministic Route Mapping**: Each distinct raw parameter value is assigned to a fixed combination of shared rails. If two parameters in a tensor share the exact same value, they are mapped to the exact same route.

## Validation Tiers

The RailNet evaluation pipeline strictly categorizes exactness into three tiers (see `railnet.verification` and Stage 12 Linear Runner):

### Tier 1: Weight Exactness (Mandatory)
Verifies that the compiled route map and shared rails exactly reconstruct the original bit-pattern of every unique weight value in the target tensor.
- **Criteria**: The `uint16` bit pattern reconstructed from $\sum_{t} \pm R_t$ matches the original `uint16` bit pattern.
- **Enforcement**: Fails compilation if a single parameter lacks a mathematically exact route.

### Tier 2: Mathematical Output Exactness (Mandatory)
Verifies that the linear operations against the RailNet topology produce mathematically identical results to the standard dense matrix multiplication.
- **Criteria**: Evaluated using the `Fraction` Oracle (no floating-point rounding errors). The rational number computed via the sparse accumulation matches the rational number computed via dense multiplication.
- **Enforcement**: Must pass for a tensor execution to be marked as "Proven".

### Tier 3: FP64/BF16 Rounded Diagnostic (Diagnostic Only)
Measures the differences caused strictly by floating-point accumulation order.
- Because `(a + b) + c` is not always exactly equal to `a + (b + c)` in standard floating-point representation, the *order* in which the sparse kernel accumulates terms compared to dense dot-products will result in minute differences.
- **Criteria**: This is a diagnostic metric only. It does not indicate a loss of exactness, merely the reality of floating-point arithmetic ordering.

## Verification API

```python
from railnet.verification.exact import verify_tensor_exact

# Verifies that every unique uint16 value maps exactly
# to its target route combination:
results = verify_tensor_exact(unique_bits, route_table, rails_array)

if results["lossless"]:
    print("Exact reconstruction proven.")
```

## Known Constraints
- The `repair_missing_values` loop in the compiler will fall back to allocating exact rails for stubborn missing values, bounded by `MAX_RAIL_REPAIRS_PER_ITER`. If an exact mapping cannot be found within the `rail_count` limit, the compilation will fail explicitly rather than quietly degrade.
