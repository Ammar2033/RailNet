# Compute → throughput (open problem #2)

Measured with `railnet.analysis.compute_cost` on the Gemma3 1B compile
(per-element `avg_terms` read from the real routing tables, ~2.3).

## Per-token arithmetic, RailNet linear vs dense

| quantity | dense | RailNet | ratio |
|---|---|---|---|
| multiplies | `out·in` | `out·rail_count` | **≈ 0.064** (93–94% fewer) |
| adds | `out·in` | `out·in·avg_terms + out·rail_count` | **≈ 2.5×** |
| total ops | `2·out·in` | ≈ `out·in·(avg_terms) + 2·out·rail_count` | **≈ 1.30×** |
| weight-sized memory read | `out·in` × BF16 | `out·in` × uint16 route-id | **1.0×** |

So RailNet **converts a balanced multiply/add workload into a mostly-add
workload**: ~15× fewer multiplies, ~2.5× more adds, ~1.3× more total ops, and
the same weight-sized memory traffic. "93% fewer multiplies" is real; it is
**not** a throughput or energy number (spec 11).

## When this is a win

Only if, in the target technology, `removed_multiplier_cost > added_adder_cost
+ routing_fabric_cost`:

- **FPGA.** DSP48 blocks are a scarce fixed resource; adders are cheap LUT
  fabric. Trading 15× multiplies for 2.5× adds frees DSPs — potentially many
  more linears per device. This is the strongest form of the RailNet case and
  is what `RailNet-256HX` would test.
- **ASIC.** A BF16 multiplier is ~3–4× an adder in area/energy. 15× fewer
  multipliers vs 2.5× more adders is roughly datapath-neutral to favourable —
  but the routing/gather fabric is unmodelled and is the open cost
  (`routing_storage.md`: no storage win, so routing must pay for itself on
  area/power/latency).
- **CPU/GPU.** Memory-bound and multiply/add symmetric → RailNet is ~1.3×
  *slower* in ops with no memory saving. The current NumPy runtime is
  correctness-first and not the target.

## Unmodelled (next)

- Gather / index-decode cost for the sign-weighted accumulation.
- Accumulator width, reduction-tree depth, deterministic-order constraint.
- Rail-value fetch and the routing-table lookup.
- Batch > 1: the rail reduce `Σ_r R_r·G[j,r]` amortizes rail-value reuse across
  tokens; the accumulation does not.

Status: `RESEARCH`. No throughput claimed until an RTL/FPGA measurement exists.
