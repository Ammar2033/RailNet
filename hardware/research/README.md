# RailNet hardware research

Status: **RESEARCH / no silicon.** Nothing here is a decision. The point of
this directory is to answer, with measurements, whether the FPGA program
(`RailNet-256HX`, roadmap Phase 8+) is worth starting.

## What is measured (Gemma3 1B, real compile)

| Question | Tool | Result |
|---|---|---|
| Does the route map compress? | `railnet.analysis.route_compression` / `routing_storage.md` | **No meaningful win.** Best lossless ≈ 0.79× dense; per-block palette and RLE are *worse* than dense. The route map is ~`log2(unique_routes)` ≈ 12 bits of real entropy per element with little spatial structure. |
| Does the compute drop? | `railnet.analysis.compute_cost` / `compute_throughput.md` | **Multiplies −93.6%, adds +2.5×, total ops +1.3×, weight memory ×1.0.** RailNet turns a balanced mul/add workload into a mostly-add one. Not a speedup by itself. |
| Is the graph correct? | `research/crosscheck_hf.py` | RailNet dense path matches HuggingFace Gemma3 (argmax + greedy identical, cosine 0.99999996). rail ≡ dense is separately proven. |

## The FPGA case, stated honestly

RailNet does **not** win on weight storage and does **not** win on total
arithmetic or memory bandwidth. The only lever is:

> On an FPGA, hard multipliers (DSP48) are a scarce fixed resource and adders
> are cheap LUT fabric. Trading ~15× fewer multiplies for ~2.5× more adds could
> fit many more linears per device — *if* the routing/gather fabric is cheap.

That "if" is open problem #3 and it needs RTL, not analysis:

- routing fabric area/power/latency vs a dense MAC array (`routing_storage.md`
  shows the *bits* are ~dense, so the fabric must earn its place on PPA)
- accumulator width + reduction-tree depth + the deterministic-order constraint
  (`accumulator.md`)
- gather / index-decode cost for the sign-weighted accumulation (unmodelled)
- rail-value storage + the routing-table lookup (`rail_primitive.md`)
- shared-multiplier count sweep 8/16/32/64/128/256 + time-mux (§22, §54)

## Recommended next steps

1. Finish the full Gemma compile, run the full `route_map_study.py` /
   `compute_cost` (attention vs MLP split).
2. Analytical FPGA resource model: for one `rail_linear` at `N` shared
   multipliers, estimate DSP / LUT / FF / BRAM and cycles, vs a dense
   equivalent. No board yet — just the datapath.
3. Only then: Phase-1 RTL (`rail_bank` + Python golden reference).

Do **not** pick a board, write PCIe/DMA, or open GDSII discussions before
step 2 gives a datapath that looks favourable.
