# Reprogrammable weight-in-silicon (the RailNet-vs-Taalas thesis)

Status: `RESEARCH`. This is the architecture the project's differentiator
implies; the numbers below are first-order and need RTL + a real NVM density
figure.

## The claim

Weight-in-silicon inference (Taalas etches weights into fixed ROM: one chip per
model) wins by killing weight-fetch energy — the dominant cost of inference.
RailNet's bet: get most of that win **while staying reprogrammable**, so one
fabric serves many models.

## Why not just put raw BF16 weights in rewritable NVM?

You could. But then each compute lane needs a full BF16 multiplier, and the MAC
array is a large fraction of tile area. RailNet's rail decomposition means the
weights-in-NVM are 16-bit **route ids**, and the compute is:

    G[j,r] = Σ_i sign(i,j,r)·X[i]      # adds only (sign is ±1)
    Y[j]   = Σ_r R_r · G[j,r]          # only rail_count (=96) real multiplies

So a RailNet tile needs **~12× fewer hard multipliers** than a dense tile at
matched throughput (`fpga_resource_model.md`), at the cost of ~2.5× more adders
(cheap) and a routing/gather structure (unmodelled — this is the risk).

Smaller compute tile → more die area goes to NVM → **denser effective
model storage per mm²** than raw-weights-in-NVM + dense MACs.

## Tile sketch (digital, weight-stationary)

```
   route-id NVM bank  (out_tile × in × 16 bit, ReRAM/MRAM)
        │
   route decode       (id → up to max_terms (rail_idx, sign))
        │
   stage-A adder tree  (± X[i] into G[j, r];  ~in·avg_terms adds / output)
        │
   G accumulators      (rail_count per active output j)
        │
   stage-B rail MAC    (Σ_r R_r·G[j,r];  M shared BF16 multipliers, M ≪ in)
        │
   Y[j]
```

`R` (96 × BF16) and the routing table (per-tensor, ~KB) sit in registers / a
small SRAM and are the reprogrammed-per-model state alongside the NVM route ids.

## First-order storage feasibility (Gemma3 1B linears)

- route-id map ≈ 1.25 GB = 10 Gbit (≈ dense weights; no compression —
  `routing_storage.md`, ADR 0001)
- ReRAM density is process-dependent, ~10–100 Mbit/mm² → **100–1000 mm²** for a
  1B model. Same regime as Taalas: a single die suits small models; larger
  models need multi-die (§16, §70).
- SRAM alternative: ~1.25 GB SRAM ≈ 3–5× the die area → not competitive.

## Reprogrammability premium

- ReRAM/MRAM write endurance ~10⁶–10⁹ cycles, write time ~10–100 ns. "Load a
  model, run it for days/weeks" needs almost no endurance.
- Multi-model serving TCO: fixed-ROM = one mask set + one wafer run **per
  model**. RailNet = one, reprogrammed. For a fleet running N models this is
  N× NRE and N× inventory vs 1×.

## Open questions (in priority order)

1. **Routing fabric area/power/latency** — the stage-A gather (route-id → which
   `X[i]` into which `G[j,r]`). No bit-count help (`routing_storage.md`). Needs
   an RTL sketch of one small tile vs a dense tile. **This is the gate.**
2. Real ReRAM/MRAM density + endurance from a current PDK / literature.
3. Analog CIM kills RailNet's multiplier advantage (the crossbar multiplies for
   free) — this thesis is **digital** weight-stationary only. Confirm.
4. G-accumulator banking, rail-value fanout, deterministic order
   (`accumulator.md`).
5. Does structured / bounded-error routing (ADR 0001, option A′) shrink the NVM
   enough to matter — or is the ~12-bit/weight entropy floor real
   (`routemap_ceiling.py`).
