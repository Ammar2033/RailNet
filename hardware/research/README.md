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

## Decision (ADR 0001): path B

Route-map compression is **dead** for normally-trained weights — a 16×16 block
touches 56–67 of 96 rails (no locality); structured routing costs ≈ 2× dense.
So the route-id map is stored **dense in rewritable NVM**, and RailNet's value
is entirely the ~12× smaller compute tile.

## Recommended next steps

1. **RTL sketch of stage A** for one small tile (64×64) — the sign-weighted
   gather is the one unmodelled cost and the gate for the whole ASIC path. Real
   LUT / FF / Fmax, verified against `railnet.verification`. If it is not
   clearly cheaper than a dense MAC array → fall back to option C.
2. **ReRAM / MRAM density + endurance scan** — size the route-id NVM for a 1B
   model (`reprogrammable_weight_in_silicon.md`).
3. **Parallel research: train in the rail basis** — the only route back to a
   storage win (`docs/adr/0001`, parallel bet). nanoGPT-class prototype:
   constrain weights to `Σ sign·rail` during training + a route-map
   compressibility regulariser; measure natural route-map entropy vs baseline.
4. Then a full Phase-1 RTL block (`rail_bank` + stage A + stage B) for one tile.

Still do **not** pick a board, write PCIe/DMA, or open GDSII discussions before
step 1 gives a real gather cost.
