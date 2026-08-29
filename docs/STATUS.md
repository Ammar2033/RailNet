# RailNet — status & strategy (living)

Last updated 2026-08-29.

## Proven

- **Lossless BF16 rail representation.** 182/182 Gemma3 1B linear tensors compile
  exact via the rail-count escalation ladder — 175 at 96 rails, 6 at 128, 1 at 192.
  Reproducible: `railnet compile model_data/model.safetensors --resume`.
- **rail path ≡ dense path** of the same transformer graph, BF16-bitwise —
  **verified on the full Gemma3 1B**: 26/26 layer hidden states exact, 0/262144
  logit-bit mismatch, greedy generation identical (`results/gemma_repro.json`).
  Reproduce: `python research/reproduce_gemma.py --skip-compile --lean`.
- **Graph faithful to Gemma3.** RailNet's dense path vs HuggingFace `transformers`:
  argmax + greedy sequence identical, cosine 0.9999996 (`research/crosscheck_hf.py`).
- Working CPU runtime (`RailNetModel`), compiler (`compile_model`), verification
  hierarchy, and an honest analysis framework (`railnet.analysis`).

## Measured (Gemma3 1B)

| question | number | source |
|---|---|---|
| route-map storage vs dense | **1.003×** (min-width 0.80×) | `representation_cost` |
| lossless route-map codecs | zlib 0.80×; block/RLE > 1× | `route_compression` |
| structured routing (rails per 16×16 block) | **56–67 of 96** → cost ≈ 2× dense | `structured_routing_probe` |
| multiply reduction | **93.6%** | `compute_cost` |
| add / total-op ratio | 2.5× / 1.30× | `compute_cost` |
| weight-sized memory traffic | 1.0× | — |
| FPGA datapath at matched throughput | **~12× fewer DSPs, ~2.4× more LUT** | `fpga` (analytical) |
| stage-A gather, real synthesis (yosys) | **~7× fewer DSPs, ~5× more LUT**; gather ≈ 1.3× a dense MAC in LUT, **0 DSP** — *if G is in memory, not a register file* | `hardware/rtl/` |
| train in rail basis — loss | matches dense (0.11 vs 0.12) | `train_rail_basis` |
| train in rail basis — route-map compressibility | **none**, even with a hard 4-term budget + entropy + L1 (~90% of weights keep a unique route; 15/16 bits) | `train_rail_basis` |

## The strategic picture

RailNet's differentiator vs weight-in-silicon (Taalas): **reprogrammable** —
one fabric, many models. For that to beat a fixed-ROM chip the on-chip model
must fit in a rewritable medium.

- **It does not shrink the model.** The route-id map ≈ the weights and does not
  compress — proven three ways. Trained transformer weights carry ~12 bits/weight
  of genuine entropy with no rail locality.
- **Its only real lever is compute.** ~12× fewer hard multipliers → a small
  compute tile. The feared cost — the stage-A routing/gather fabric — **synthesised
  cheap**: ~1.3× a dense MAC in LUT, 0 DSP, *provided the G accumulator is a
  memory not a register file* (a register file is 41× — the wrong design). Net
  per column ≈ 7× fewer DSPs for ≈ 5× more LUT.
- **Total energy is ~neutral.** Inference is memory-bound; RailNet doesn't cut
  weight-sized memory traffic.

## Decision (ADR 0001)

**Path B** — route-id map stored dense in on-chip rewritable NVM (ReRAM/MRAM);
value is the small shared-rail compute tile.
**Gate: provisionally PASSES.** `hardware/rtl/` — Amaranth stage-A tiles,
functionally verified vs a golden model, synthesised with yosys. The gather is
~1.3× a dense MAC in LUT with 0 DSP (G in memory). Still unverified before the
"provisionally" drops: BF16 datapath, Fmax (no P&R), route-id decode logic,
stage-B amortisation. See `hardware/research/stage_a_rtl.md`. If a later check
kills it → fall back to **C**.

**Parallel bet — train in the rail basis: looking closed.** The rail basis
trains to dense loss, but the route-map does not compress even with a hard
4-term budget + entropy + L1 pressure — ~90% of weights keep a unique route.
A linear layer wants ~`out·in` independent numbers and freely uses the ~10⁵
available routes; there is no loss incentive to reuse them. A route-sharing VQ
(force weights onto K learned prototype routes) is the last idea; the natural
tendency is strongly against it. **Provisional read: the storage-compression
path is closed. RailNet's on-chip story is dense route-map in dense NVM.**

## Honest venture read

- **ASIC venture:** the RTL gate provisionally passed — the routing fabric is
  cheap, so the "reprogrammable weight-in-silicon" thesis is *alive*. Still
  neutral on model footprint and total energy/token; the pitch is small models,
  reprogrammable, DSP-lean — a real but narrow segment (fleets serving many
  models). Contingent on BF16/Fmax follow-ups.
- **FPGA niche:** legitimate and now gate-backed — ~7× fewer DSPs (synthesised)
  for DSP-bound edge inference, working software today.
- **Research contribution:** strong and real — a verified, HF-faithful exact
  neural execution architecture + an honest, measured analysis (and RTL) of when
  weight-in-silicon helps.

## Next

1. **BF16 datapath** for the tiles + re-synth (does the DSP ratio hold?).
2. **Fmax** — nextpnr on ECP5 for the stage-A RMW pipeline.
3. Full tile end-to-end (route-id BRAM → decode → 3× StageA → StageB), verified
   against `railnet.verification` on a real compiled tensor.
4. ReRAM/MRAM density + endurance scan.
5. (lower priority) route-sharing-VQ train-in-rail-basis experiment.
