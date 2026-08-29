# RailNet — status & strategy (living)

Last updated 2026-08-29.

## Proven

- **Lossless BF16 rail representation.** 182/182 Gemma3 1B linear tensors compile
  exact via the rail-count escalation ladder (most at 96 rails, ~7 need 128–192).
- **rail path ≡ dense path** of the same transformer graph, BF16-bitwise, full
  vocab + per layer (`railnet.verification.verify_forward`).
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
| train in rail basis — loss | matches dense (0.12 vs 0.12) | `train_rail_basis` |
| train in rail basis — route-map compressibility | none by default (~10 rails/weight, every weight a unique route) | `train_rail_basis` |

## The strategic picture

RailNet's differentiator vs weight-in-silicon (Taalas): **reprogrammable** —
one fabric, many models. For that to beat a fixed-ROM chip the on-chip model
must fit in a rewritable medium.

- **It does not shrink the model.** The route-id map ≈ the weights and does not
  compress — proven three ways. Trained transformer weights carry ~12 bits/weight
  of genuine entropy with no rail locality.
- **Its only real lever is compute.** ~12× fewer hard multipliers → a small
  compute tile → more die for storage. On a DSP-bound FPGA this is a genuine
  advantage; on an ASIC it depends on the (unmodelled) routing fabric cost.
- **Total energy is ~neutral.** Inference is memory-bound; RailNet doesn't cut
  weight-sized memory traffic.

## Decision (ADR 0001)

**Path B** — route-id map stored dense in on-chip rewritable NVM (ReRAM/MRAM);
value is the small shared-rail compute tile.
**Gate:** an RTL sketch of the stage-A routing/gather fabric for one small tile,
verified against `railnet.verification`. Cheaper than a dense MAC array → the
"reprogrammable weight-in-silicon" thesis holds. Not → fall back to **C** (FPGA /
software contribution).

**Parallel bet — train in the rail basis.** First probe: the rail basis trains
to dense loss but does **not** compress for free. Whether forcing compression
(hard term budget + strong route-sharing pressure) is possible at acceptable
accuracy is the open research question and the only route back to a storage win.

## Honest venture read

- **ASIC venture:** not yet, and the current numbers make the thesis hard —
  neutral on both memory footprint and energy/token. Needs the RTL gate to pass
  **and** either a train-in-rail-basis breakthrough or a dense-NVM density path.
- **FPGA niche:** legitimate — ~12× fewer DSPs for DSP-bound edge inference,
  working software today.
- **Research contribution:** strong and real now — a verified, HF-faithful exact
  neural execution architecture with an honest analysis of when weight-in-silicon
  helps.

## Next

1. Stage-A routing-fabric RTL sketch (the gate).
2. ReRAM/MRAM density + endurance scan.
3. Serious train-in-rail-basis experiment: term-budget + route-sharing VQ, real
   task, loss-vs-compression curve.
4. Finish + verify the 182/182 compile; full `railnet cost` numbers.
