# ADR 0001 — On-chip model storage strategy

Status: **ACCEPTED** — path **B**, with a train-in-rail-basis bet in parallel.
Date: 2026-08-29.

## Context

RailNet's differentiator vs weight-in-silicon inference (e.g. Taalas, which
etches weights into fixed ROM — one chip per model) is **reprogrammability**:
the same physical fabric runs many models by reloading rails + routing. KV
cache and other dynamic state live in SRAM (spec 3, §28).

For that to beat a fixed-ROM chip, the on-chip *model* footprint — rails +
routing table + route-id map — must fit on a die in a **rewritable** medium.

Measured (Gemma3 1B, 175/182 linears, this session):

| part | size | note |
|---|---|---|
| rails | 96 × BF16 = 192 B / fabric | trivial |
| routing table (bits→route) | ~15–24 KB / tensor | small |
| **route-id map** | **1.25 GB** (≈ dense weights) | the problem |

The route-id map is one `uint16` per weight and it is `route_id[i,j] =
f(weight_bits[i,j])` — i.e. the weights relabeled. Lossless codecs on it:
zlib ≈ 0.8×, block-palette / RLE **worse** than dense. Effective Representation
Cost of the shipped artifact is **1.003× dense**.

So RailNet, as it stands, does **not** shrink the model. Putting it in SRAM
needs ~1.25 GB (3–5× the die area a fixed-ROM chip needs for the same model,
since SRAM is ~4–10× less dense than ROM); streaming route-ids from DRAM has the
**same bandwidth/energy as fetching dense weights** — losing the whole point.

## Options

### A — Compress the route-id map (lossless, post-compile)
Hierarchical routing, cross-tensor palette, graph factoring, entropy coding.
*Ceiling:* bounded by lossless compressibility of trained BF16 weights
(historically ~10–30%). Viable only if the ceiling study shows large
cross-tensor route sharing. Keeps strict exactness.

### A′ — Compile to a compressible map (structured / bounded-error routing)
Constrain the compiler: a block of weights shares one rail subset `S` (|S| ≤ k)
and stores only per-weight sign bits (~k bits) + a small exception list.
Map cost drops toward `k` bits/weight + template ids. Potential 4–6×.
*Cost:* gives up strict bit-exactness for a **bounded error** (target: below
BF16 rounding for ≥ 99.x% of weights). A philosophical shift for the project;
exactness stays available as a separate mode.

### B — Dense rewritable NVM (ReRAM / MRAM) + shared-rail digital compute-near-memory
Do not compress. Put the route-id map in on-chip ReRAM — ~ROM-dense **and**
rewritable. A shared 96-rail multiplier bank + adder fabric computes near the
storage. The rail decomposition's value here: ~12× fewer multipliers
(`fpga_resource_model.md`) → a small compute tile → more die for storage →
denser. *Cost:* an architecture bet; needs a real ReRAM density number to size;
1B model ≈ 10 Gbit → 100–1000 mm² depending on process (same regime as Taalas).

### C — Drop the "model in silicon" pitch
Ship the FPGA angle: ~12× fewer DSPs is real and useful for DSP-bound edge
FPGA inference. A working software stack exists today. Not an ASIC venture.

## Evidence

Measured on the compiled Gemma3 1B artifacts (`railnet.analysis` +
`research/*_probe.py`):

| test | result | reads as |
|---|---|---|
| lossless codecs on the route-id map | zlib ≈ 0.80×, block-palette / RLE **> 1×** | **A dead** |
| structured routing — distinct rails per 16×16 block | **56–67 of 96** (a whole row: 71–84) | **A′ dead** — no rail locality; structured cost ≈ **2× dense** |
| Effective Representation Cost, as shipped | **1.003× dense** | route map ≈ weights, no compression |

Trained transformer weights carry ~12 bits/weight of genuine entropy with **no
exploitable block structure** in their rail assignment. Post-compile
compression cannot fund a "model in SRAM" story.

## Decision

**Path B.** The route-id map is stored **dense** in on-chip rewritable NVM
(ReRAM / MRAM — ~ROM density, rewritable). RailNet's entire value is on the
compute side: the rail decomposition needs ~12× fewer hard multipliers
(`fpga_resource_model.md`), so the compute tile is small and more die goes to
NVM → denser than raw-weights-in-NVM + dense MACs, and reprogrammable unlike
etched ROM.

**Gate — provisionally PASSED.** `hardware/rtl/` (Amaranth, yosys-synthesised,
functionally verified): with the G accumulator in a memory (not a register
file — that is 41× a dense MAC), the stage-A gather is ~1.3× a dense MAC in LUT
with **0 DSP**. Per output column ≈ 7× fewer DSPs for ≈ 5× more LUT. Follow-ups
before it's unqualified: BF16 datapath, Fmax (P&R), route-id decode logic. If
one of those kills it → fall back to **C**. See `hardware/research/stage_a_rtl.md`.

**Parallel bet — train in the rail basis: probed, provisionally closed.**
`research/train_rail_basis.py` (tiny char-LM): the rail basis trains to dense
loss, but even with a hard 4-term budget + entropy + L1 the route-map does not
compress — ~90% of weights keep a unique route, 15/16 bits. A linear layer
freely spends the ~10⁵ available routes and the loss gives no reason to reuse
them. A route-sharing VQ is the only remaining idea. Barring that, **the
storage-compression path is closed** and the NVM in path B holds the full
dense route-map.

## Consequences

- Hardware research reprioritises: **routing-fabric RTL** (open problem #3) and a
  **ReRAM/MRAM density scan** move ahead of PCIe / board selection.
- A `research/` train-in-rail-basis track opens (needs torch; installed).
- Strict-exact compile stays the verification / "PROVEN" mode and the
  credibility anchor. A `bounded-error` mode is only worth building if the
  train-in-rail-basis bet shows promise.
- The public framing drops any implied "runs 32B on one chip via compression";
  the honest pitch is "reprogrammable weight-in-silicon, dense NVM, small shared
  compute" — feasible for small models on one die, multi-die beyond.
