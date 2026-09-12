# RailNet — Autonomous Engineering Loop Log

Newest entry first. One entry per loop turn: what changed, what was run, the
numbers, what failed, what risk remains, and the single most valuable next job.
Technical report register — no marketing language, no claim promoted above the
evidence behind it.

---

## 2026-09-12 — Local turn: Gate 3 closed, Gate 4 quantified

**Base:** `97633db` → **head:** `71b0b02`. Run locally (the scheduled cloud
routine is paused: the Claude GitHub App has read-only access to the repo, so
three overnight turns did real work and could not push any of it).

### Changed

| File | Change |
|---|---|
| `hardware/fpga/build_fpga.py` | nextpnr's `SystemExit` is caught and classified; `_derive_pnr_status()` derives status from exit code, `ERROR:` lines and routing completion; failed runs publish no Fmax; dry runs write to a separate file; `main()` exits non-zero on failure; evidence tier corrected from Tier 6 to Tier 5 |
| `scripts/verify_all_claims.py` | prints what the run observed instead of a stored string; `hardware_verified` pinned `False`; CLM-05, CLM-09, CLM-10 rewritten against measurement |
| `benchmarks/benchmark_fpga_prototype.py` | `load_routed_fmax()` can now fail instead of inventing 83.79; clock provenance stamped into output; removed a hardcoded bit-exact claim printed as `PASS (100%)` |
| `hardware/asic/synth_asic.py` | real `dfflibmap` + `abc -fast -liberty` mapping; runs native Yosys in the OpenLane container (staging into temp because Docker cannot mount this drive); area read from `stat -liberty -json`; honest `BLOCKED` path when no Liberty |
| `hardware/asic/gate3_evidence.json`, `gate4_memory_area.json` | new, committed on purpose — `build/` and `results/` are gitignored |
| `tests/unit/test_asic_synth.py` | new, 9 tests |
| `tests/hardware/test_physical_pcie_readiness.py` | regression test that a failed P&R is not reported as a completed run |
| `docs/PROJECT_REALITY_AUDIT.md` | new — the Phase 0 deliverable, missing until now |
| 5 documents | 83.79 MHz corrected wherever it was presented as measured |

### Measured

- **Gate 3 (new):** `railnet_top` → **39,706 `sky130_fd_sc_hd__*` cells (55 types),
  245,247.712 µm² (0.245 mm²)**, 24 `$mem_v2` left unmapped. Native Yosys 0.38,
  `abc -fast`, ~11 s. First real standard-cell mapping in the project's history.
- **Gate 4 (new):** memories as flip-flops → 317,850 cells / 2,309,721.456 µm².
  Memories are **9.4× the logic area**; real macros would displace **≈2.06 mm²**.
- **ECP5 re-run:** `stagea_bram` 209.12 MHz (0 DSP), `stageb_int8` 157.80 MHz,
  `full_int8_tile` 95.50 MHz. `railnet_top_2x2` **does not route**.
- Full test suite: exit 0 after every change. New ASIC tests: 9/9.

### Failed / abandoned

- `yowasp-yosys` (WASM) cannot complete standard-cell mapping: per-module it
  re-parses the 12.8 MB Liberty (20+ times), flattened it ran 40+ minutes with
  no result. Abandoned in favour of native Yosys in the container.
- Two of my own bugs cost a run each: the Liberty referenced by absolute host
  path inside the container, and the area read from a log line Yosys 0.38 never
  prints. Both now covered by tests.

### Risks remaining

1. **The Gemma3 1B exactness claim does not reproduce on this host.** Correcting
   myself twice here. I first wrote that the weights were missing; they are not —
   `model_data/model.safetensors` is a real 2 GB file. The reproduction was
   unrunnable for a different reason: `compiled/manifest.json` records
   `/Volumes/SSD/…` macOS absolute paths and `_resolve()` trusted any absolute
   path without checking it existed, so `RailNetModel.load()` failed on every
   machine but the original author's. With that fixed (and a regression test
   added), the run completes and **fails**: `verify_compiled` PASS 182/182,
   `verify_generation` PASS, but `verify_forward` reports
   `logit_bf16_mismatch: 281`, `all_layers_exact: false` against a documented
   claim of 0/262144.
   The cause is not a wrong checkpoint: `routeids` matches the source bf16 bits
   element-for-element across 3 tensors in 2 layers (~17 M elements, 0
   mismatches), so the artifact was compiled from exactly these weights.
   The profile points at drift rather than breakage: `first_divergent_layer: 2`
   (layers 0 and 1 exact), then 20 of 26 layers off by only 2–6 bf16 values
   each, ending at 281 of 262,144 logit bits (~0.1%), with greedy generation
   still token-identical. A wrong checkpoint or a broken kernel would diverge at
   layer 0 and grow fast; this does not.
   What remains open is whether it is same-graph drift or a platform float
   difference — the original run was almost certainly macOS, this is
   Windows/NumPy. The cheap decider is the Fraction oracle
   (`docs/EXACTNESS.md` Tier 2) on one diverging layer: exact in rational
   arithmetic means the platform is responsible.
   **Do not restate the 0-mismatch claim until then.**
2. **ReRAM density, energy and TCO remain assumptions.** The entire
   "reprogrammable weight-in-silicon" thesis rests on them and no PDK or IP
   backs them.
3. **The 83.79 MHz grid figure is dead** until the LPF constrains the AXI pins
   or the grid is timed out-of-context.
4. The scheduled cloud loop cannot persist anything until the GitHub App is
   granted write access.

### In flight at time of writing

OpenLane flow on `railnet_top` at 20 ns (Gates 5/6/7). Note it is placing the
**flip-flop** variant — 317,850 cells, 2.31 mm² — because no macro is wired in,
so congestion trouble would itself be evidence for Gate 4.

### Next most valuable job

**Wire the four 16b×1024 route memories to `sky130_sram_1kbyte_1rw1r_8x1024_8`
macro pairs.** They are the only clean geometric fit in the PDK and they carry
most of the 2.06 mm² the memories currently cost, so this is the single change
that most improves the design's physical reality — and it unblocks a meaningful
Gate 5 run instead of placing a quarter-million flip-flops.
