# RailNet hardware research

Status: **RESEARCH / no silicon.** Nothing here is a decision. The point of
this directory is to answer, with measurements, whether the FPGA program
(`RailNet-256HX`, roadmap Phase 8+) is worth starting.

## What is measured (Gemma3 1B, real compile)

| Question | Tool | Result |
|---|---|---|
| Does the route map compress? | `railnet.analysis.route_compression` / `routing_storage.md` | **No meaningful win.** Best lossless ≈ 0.79× dense; per-block palette and RLE are *worse* than dense. The route map is ~`log2(unique_routes)` ≈ 12 bits of real entropy per element with little spatial structure. |
| Does the compute drop? | `railnet.analysis.compute_cost` / `compute_throughput.md` | **Multiplies −93.6%, adds +2.5×, total ops +1.3×, weight memory ×1.0.** RailNet turns a balanced mul/add workload into a mostly-add one. Not a speedup by itself. |
| Does the FPGA datapath favour it? | `railnet.analysis.fpga` / `fpga_resource_model.md` | At matched throughput: **~12× fewer DSPs, ~2.4× more LUT** (analytical). Favourable on a DSP-bound device — *if* stage A's gather is cheap (unmodelled). |
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

1. ~~Analytical FPGA resource model~~ — done (`fpga_resource_model.md`): the
   datapath trade is favourable (~12× fewer DSPs).
2. **RTL sketch of stage A** for one small tile (e.g. 64×64) — the sign-weighted
   gather is the unmodelled cost. Get a real LUT / FF / Fmax number for it,
   verified against `railnet.verification` as the golden reference.
3. Then a full Phase-1 RTL block (`rail_bank` + stage A + stage B) for one tile.
4. Finish the full Gemma compile; run `route_map_study.py` / `compute_cost` with
   the attention-vs-MLP split for completeness.

Still do **not** pick a board, write PCIe/DMA, or open GDSII discussions before
step 2 gives a real gather cost.
