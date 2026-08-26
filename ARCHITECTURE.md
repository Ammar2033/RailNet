# ARCHITECTURE — RailNet

## 1. Rail Representation

Rail = dtype-independent shared primitive numeric value.
```
R0 = +0.03149 (BF16 bits 0x3D08 ...)
W1 = +R3 - R17 + R42
```
`railnet/rails/rail.py` — `Rail(id, dtype, encoded_value, decoded_value)`; `RailBasis` is ordered fabric `R0..R95`.

Physical rail capacity fixed; values programmable per artifact (model-specific config).

## 2. Topology

Per-weight route: `parameter index → [(rail, sign), ...]`
No per-weight floating coefficient. Stored as `bf16_bits → route` dictionary + per-element `route_ids` map (uint16).

`railnet/topology/route.py` · `graph.py` · `scheduler.py`

## 3. Shared Computation

Dense: `Y[j]=Σ_i X[i]·W[i,j]`
RailNet: `W[i,j]=Σ_r sign·R_r` → `Y[j]=Σ_r R_r · Σ_i sign·X[i]` — bincount over `p_idx`.

Proven shared-multiply reduction ≥93% (full Gemma3 1B).

## 4. DType Abstraction

`RailDType` interface (`encode/decode/quantize/exact_equal`). Registry: `bf16` PROVEN, `fp16/fp32/int8/int16` READY, `int4` PLANNED.

## 5. Compiler

`RailNetCompiler.compile_tensor(raw, dtype, rails, max_terms)` — generic entry.
BF16 backend: weighted quantile init → coordinate-descent `update_basis` → exhaustive exact routing (`greedy` for speed, `exhaustive` for verdict) → `repair_missing_values`/`repair_safe_slots`. Escalation ladder 96→128→192.

Per-model adapters: `GemmaAdapter` PROVEN, `Llama/Qwen` PLANNED.

## 6. Artifact

`model.rnmodel` — `RNET` magic, version, dtype, tensor manifest, rail tables, topology, SHA-256. No dense weights. Route maps as sidecar `.npy` or embedded blobs. Verified via `railnet/artifacts/reader.py`.

## 7. Runtime

Device abstraction: `RailNetDevice.cpu() / .gpu() / .pcie("railnet0")` — same semantics.
`railnet/runtime/linear.py` — `rail_linear` / `rail_linear_fast` (bincount, bit-identical order).
`transformer.py` — Gemma3 ops (`rms_norm`, `rope`, `gelu_tanh`, `block_forward`) shared by dense + rail paths.
`kv_cache.py` — per-layer growing K/V; `generation.py` — greedy deterministic loop.

## 8. Memory Model

STATIC: rails, topology, routing config, metadata.
DYNAMIC: activations, Q/K/V, attention scores, KV cache, tmp buffers.
Honest accounting in `railnet/storage/` — route map often dominates.

## 9. PCIe Architecture (research)

```
HOST → SDK/Runtime → DMA/command queue → PCIe → RailNet Card
  [ Controller | Rail Fabric | Routing Fabric | Shared Compute | Activation/KV Memory ]
```
Software-first: CPU simulator → virtual PCIe device → driver/DMA model → FPGA → ASIC.

## 10. Verification

`railnet/verification/` — `WeightReconstructionOracle`, `LinearKernelOracle`, `TransformerOracle`, `GenerationOracle` with BF16-bitwise exactness.

## 11. Limitations & Next

Route-map compression, minimum exact rail count, global model basis, multi-model fabric, INT8/FP16 support, hardware rail/routing primitives, KV cache architecture, PPA — all under `hardware/research/` and `railnet/topology_compression/`.
