# SPEC — RailNet Technical Rules

## 1. No hidden dense weight array
Artifact and runtime MUST NOT contain or require dense `W` at inference. Verified by `runtime_weight_array: ABSENT`.

## 2. Exactness
*Weight exactness*: `float32_to_bf16_bits(reconstructed) == target_bits` for every unique value when `exact=True`.
*Logit exactness*: BF16-bitwise equality over full vocab × sequence (1,048,576 logits in Gemma3 1B stage 15B).
Report as `exact/total`; lossless iff `exact==total`.

## 3. Static vs Dynamic State
Static (artifact): rails, topology, schedules, metadata, checksums.
Dynamic (runtime): activations, hidden, Q/K/V, attention probs, KV cache, tmp buffers.

## 4. Artifact Validity
Magic `RNET`, version, dtype, shape, rail_count, topology schema; SHA-256 over canonical JSON; model/tensor hashes. Mismatch → runtime MUST NOT start.

## 5. DType Semantics
All dtypes via `RailDType` (`encode/decode/quantize/exact_equal`). Unsupported dtype → `NotImplementedError`, never silently fall back.

## 6. Topology Semantics
`W = Σ sign·R[rail]` — sign ∈ {+1,−1}, no coefficient, distinct rails per weight, `max_terms` is upper bound.

## 7. Runtime Semantics
* Correctness > memory safety > determinism > throughput.
* Accumulation order for `rail_linear_fast` bit-identical to `rail_linear`.
* Deterministic generation (greedy, temperature 0) for verification.

## 8. Memory Accounting
Always report honest full representation: `rail_bits + route_map_bits` vs `dense_bits`. Dictionary-only bits are invalid.

## 9. Benchmark Definitions
Correctness: dense reference vs RailNet logits exactness.
Performance: wall-clock of rail vs dense CPU (oracle time excluded); future GPU/PCIe measured separately. No ASIC/FPGA throughput claimed until hardware exists.

## 10. Status Labels
Every feature: `PROVEN | EXPERIMENTAL | PLANNED`. Do not promote experimental to proven.
