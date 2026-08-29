# Rail primitive — research options

A "rail" is a **programmable shared numeric primitive**, not (yet) a physical
multiplier. `rail_count = 96` for the Gemma3 1B compile; values are loaded per
artifact via DMA. One logical rail is not necessarily one physical multiplier
(§6, §24).

## Storage options for the rail values

96 × BF16 = 192 bytes per fabric — tiny. Options: register bank, distributed
RAM, small SRAM, DSP coefficient path. This is not a bottleneck; the routing
table and route-id map are (`routing_storage.md`).

## Compute options

`Y[j] = Σ_r R_r · G[j,r]` where `G[j,r] = Σ_i sign(i,j,r)·X[i]`.

- The `Σ_r R_r · G[j,r]` step is `out · rail_count` real multiplies — this is
  where a small pool of shared BF16 multipliers (8/16/32/…, time-multiplexed
  over the 96 rails) lives. Sweep in the FPGA resource model.
- The `Σ_i sign·X[i]` step is adds only (sign is ±1). This is the 2.5×
  add-inflation from `compute_throughput.md` and needs an adder tree + the
  routing that selects which `X[i]` land in which `(j,r)` accumulator.

## Open

- shift/add decomposition vs a real multiplier for the rail-reduce step
- fixed-point vs BF16 internal accumulation (must stay bit-exact to the
  float64 reference — `accumulator.md`)
- whether rail values can be shared across *tensors* (global fabric) or must be
  per-tensor — affects reconfiguration cost

Status: `RESEARCH`. No decision.
