# FPGA datapath resource model (open problem #3, step 2)

`railnet.analysis.fpga` — **analytical, not synthesis.** One question: to match
a dense MAC tile's throughput, what does the RailNet `rail_linear` datapath
cost in DSP / LUT / FF?

## Model

Dense tile: `P` BF16 MAC units (1 DSP each), `ceil(out·in / P)` cycles/token.

RailNet, two pipelined stages over the output stream:
- **A** — sign-weighted accumulation `G[j,r] = Σ_i sign·X[i]`: `out·in·avg_terms`
  ± adds, `A` fabric adders, `ceil(out·in·avg_terms / A)` cycles.
- **B** — rail reduce `Y[j] = Σ_r R_r·G[j,r]`: `out·rail_count` MACs, `M` DSPs,
  `ceil(out·rail_count / M)` cycles.

Cycles/token = `max(A_cycles, B_cycles)` (pipelined). `A` and `M` are sized to
the dense tile's cycle budget.

## Result (Gemma3 1B compile, `avg_terms ≈ 2.3`, `rail_count = 96`)

One `1024×1152` linear, dense tile = 64 DSPs:

| | DSP | fabric adders | LUT (glue) |
|---|---|---|---|
| dense | 64 | 0 | ~3.0k |
| RailNet | **6** | 149 | ~9.2k |

Aggregate over 134 compiled linears:

| ratio (RailNet / dense) | value |
|---|---|
| **DSP** | **≈ 0.083  (≈ 12× fewer)** |
| LUT | ≈ 2.4× |
| cycles/token | 1.0 (matched) |

The ratio is ~scale-invariant across dense tile sizes 16–256 DSPs.

## Reading

On a **DSP-bound** FPGA this is favourable: ~12× fewer DSPs for ~2.4× more LUT
(abundant) means far more linear compute per device — *if* stage A's routing is
cheap.

## The unmodelled cost (this is the real Phase-8 question)

Stage A's 149 adders each need, per cycle:
- route-id → `(rail_idx, sign)` decode (cheap, in the model)
- a **gather**: pick the right `X[i]` and target the right `G[j,r]` accumulator

That gather is a crossbar-like structure and it is **not modelled**. Given
`routing_storage.md` (the route map is ~dense, no compression), the routing
fabric has no bit-count help — it must pay for itself on area/power/latency.
The next step is an RTL sketch of stage A for one small tile to get a real
LUT/FF/timing number for the gather.

Also unmodelled: G-accumulator SRAM/FF banking, rail-value fetch, BRAM for the
routing table, deterministic accumulation order (`accumulator.md`), batch > 1.

Status: `RESEARCH`. The datapath trade looks favourable enough to justify an
RTL sketch of stage A; it does **not** yet justify board selection or PCIe work.
