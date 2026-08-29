# Stage-A routing fabric — RTL sketch + synthesis (the ADR 0001 gate)

`hardware/rtl/` — Amaranth 0.5 tiles, functionally verified against a numpy
golden model (`test_tiles.py`), synthesised with yosys `synth_xilinx`
(UltraScale+ mapping). Fixed-point int16 activations / int32 accumulator.

## What was built

| module | job | throughput |
|---|---|---|
| `DenseInner` | `acc += x·w` streaming inner loop | 1 input / cycle |
| `StageA` | `G[rail] += ±x`, G as a 96×int32 **register file** + 96-way switch | 1 term / cycle |
| `StageABram` | same, G in a **1R1W memory** (RMW pipeline) | 1 term / cycle |
| `StageB` | `Y = Σ_r rail[r]·G[r]`, one shared multiplier | 1 rail / cycle |

## Synthesis (yosys 0.68, `synth_xilinx -flatten`)

| tile | DSP | FF | LUT | vs dense (LUT-equiv) |
|---|---|---|---|---|
| dense inner | **1** | 33 | 34 | 1× |
| stage-A, register file | 0 | 3072 | 4568 | ~41× |
| **stage-A, G in memory** | **0** | **65** | **157** | **~1.3×** |
| stage-B (1 mult) | 2 | 97 | 98 | — |

**The G accumulator must be a memory, not a register file.** The naive
register-file + 96-way write demux is 41× a dense MAC — that was the feared
outcome. With G in a 1R1W memory (LUTRAM here; a BRAM at scale) the stage-A
gather is **~1.3× a dense MAC in LUT and uses zero DSPs**.

## Per-output-column trade at matched throughput

Stage-A runs at 1 term/cycle vs dense 1 input/cycle, so rail needs
`ceil(avg_terms) ≈ 3` stage-A instances to keep up, plus stage-B amortised over
~16 columns:

| | dense | RailNet |
|---|---|---|
| DSP | **1** | ~0.13 (stage-B share) |
| LUT | ~34 | ~470 |
| FF | ~33 | ~200 |

**≈ 7× fewer DSPs, ≈ 5× more LUT.** This confirms the analytical
`fpga_resource_model.md` estimate (~12× DSP, ~2.4× LUT) in the right direction
and magnitude — with real gates behind it.

## Verdict: the gate provisionally PASSES

The routing/gather fabric is **not** the dealbreaker. RailNet trades a scarce
resource (DSP) for abundant ones (LUT + distributed RAM), ~7× in RailNet's
favour on DSPs. On a DSP-bound device that means substantially more linear
compute per chip.

## What is still unverified (before "PASSES" loses the "provisionally")

1. **BF16 datapath.** int16 here. Dense keeps its multiplier (~1 DSP); stage-A's
   adder becomes a BF16 adder (~2–3× the int adder, still tens of LUT). The DSP
   ratio should hold; confirm.
2. **Fmax.** No place-and-route. The LUTRAM RMW pipeline may not reach a DSP's
   ~700 MHz — a real throughput risk. Needs nextpnr / vendor P&R.
3. **Route-id fetch + decode** (route_id → up to 4 `(rail, sign)`): not in the
   tile yet, ~20–50 LUT.
4. **Stage-B amortisation** (the ÷16) is assumed, not architected.
5. **G-memory pressure at scale.** 96×32b per active output tile — a BRAM holds
   several; fine sequentially, needs a tile-count budget.
6. **BRAM RMW hazard.** Distinct rails per weight (spec §6) avoids it within a
   weight; a controller inserts one bubble between weights (minor throughput hit).

## Next

- BF16 adder variant + re-synth.
- Full tile: route-id BRAM → decode → 3× `StageABram` → `StageB`, one controller,
  verified end-to-end against `railnet.verification` on a real compiled tensor.
- nextpnr timing on an open target (ECP5) for a first Fmax.
