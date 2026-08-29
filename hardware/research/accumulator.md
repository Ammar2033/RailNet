# Accumulator — research

The accumulator is on the critical path and on the exactness contract: the
software reference, RTL, and FPGA must all produce **the same BF16 bits**
(spec 7 — `rail_linear_fast` accumulation order is bit-identical to
`rail_linear`).

## Two accumulation stages

1. **Sign-weighted, into `G[j, r]`** — `~out·in·avg_terms` adds/token
   (`compute_throughput.md`). Needs a fixed, reproducible reduction order.
   The current kernel does a single `np.bincount` over `p_idx`
   (`railnet/kernel.py`), which fixes the order by index; hardware must match
   that order or prove order-independence for the value range.
2. **Rail reduce** — `out·rail_count` MACs: `Σ_r R_r · G[j,r]`.

## Options

- local accumulator per output vs a shared reduction tree
- carry-save vs DSP-native accumulate
- FP32 (or wider) accumulation cast to BF16 at the end — RailNet runs float64
  in software; the question is the minimum internal width that still lands on
  the same BF16 bits for the real value distribution
- pipeline depth vs latency for the decode step

## Open

- Is the sign-weighted sum order-independent for BF16 inputs in the real
  weight/activation range? If yes, hardware is free to reorder for throughput.
  If no, the bincount order is part of the ABI.
- Deterministic generation (temperature 0) must hold across CPU / RTL / FPGA.

Status: `RESEARCH`. Verification harness (`railnet.verification`) is the
reference for any RTL block.
