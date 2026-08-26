# ROADMAP — RailNet

## Phase 1 — Research proof ✅
RailNet-256, topology exact, shared multiplication, real Gemma BF16 analysis.

## Phase 2 — Full software execution ✅
MLP 3/3, layer-0 global fabric, linear runtime, transformer block, KV cache, 182/182 lossless, 26/26 layers exact.

## Phase 3 — Deterministic generation 🔄
Greedy decode, tokenizer integration, stage 16 verification.

## Phase 4 — Public open-source framework 🔜 (this release)
Package structure, dtype abstraction, CLI, artifacts, verification, CI, docs.

## Phase 5 — Route-map optimization 🔜
Compression research (`topology_compression/`): templates, hierarchical, delta/RLE, graph factoring.

## Phase 6 — Multiple dtype support 🔜
FP16/INT8/FP32 proven compilers.

## Phase 7 — Multiple model families 🔜
Llama/Qwen adapters.

## Phase 8 — Optimized CPU runtime 🔜
SIMD, multithreading, profiling.

## Phase 9 — GPU backend 🔜
CUDA shared-multiply kernels.

## Phase 10 — PCIe software architecture 🔜
Virtual device, driver/DMA model.

## Phase 11 — Rail hardware architecture 🔜
Programmable BF16 register / SRAM / coefficient datapath research.

## Phase 12 — RTL prototype 🔜

## Phase 13 — FPGA prototype 🔜
PCIe endpoint + rail engine on FPGA card.

## Phase 14 — ASIC feasibility / PPA 🔜

## Phase 15 — PCIe prototype card 🔜

## Phase 16 — Multi-chip scaling 🔜 (32B → 64B → 128B)

## Versioning

v0.1 — BF16 proven (current)
v0.2 — generic dtype API
v0.3 — route optimization
v0.4 — multi-model adapters
v0.5 — optimized runtime
v0.6 — PCIe software model
v0.7 — FPGA prototype
v0.8 — RTL architecture
v0.9 — ASIC feasibility
v1.0 — hardware-ready architecture
