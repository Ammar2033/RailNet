# ADR 0001 — On-chip model storage strategy

Status: **PROPOSED** (decision pending the route-map ceiling study).
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

## Decision

Pending the route-map ceiling study (`research/routemap_ceiling.py`):

- **sharing_factor ≫ 1** (big cross-tensor route reuse) → pursue **A**, keep exactness.
- **sharing_factor ≈ 1 and entropy ceiling ~0.8×** (expected) → **A′ + B**:
  - A′ as the near-term experiment (compiler only, no hardware, high information).
  - B as the hardware target regardless — the route-map lives in dense rewritable
    NVM; A′ only sets how much NVM.
  - Keep strict-exact compile as the verification / "PROVEN" mode.
- If A′ cannot beat ~2× and no ReRAM density path closes → fall back to **C**.

## Consequences

- The `lossless` framing in `docs/EXACTNESS.md` gets a sibling `bounded-error`
  mode; the verification hierarchy must gain an error-budget tier.
- `railnet.analysis` needs a structured-routing cost model.
- Hardware research reprioritises: routing fabric RTL (open problem #3) and a
  ReRAM density literature scan move ahead of PCIe / board selection.
