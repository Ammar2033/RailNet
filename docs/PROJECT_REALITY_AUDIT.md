# RailNet — Project Reality Audit

Phase 0 deliverable of `.claude/loop.md` (§4, §18). One table mapping each
load-bearing claim to the evidence that actually exists, its status, the risk it
carries, and what would raise it.

**Audit basis:** everything below was re-derived by running things on
2026-09-11/12, not by reading the documents. Where a claim could not be
reproduced, it says so.

| | |
|---|---|
| Base commits | `97633db` → `dde44d8` → `a804031` → `71b0b02` |
| Host | Windows 11, Python 3.14, no FPGA board, no commercial PDK, no silicon |
| EDA (native) | Yosys 0.38, OpenROAD, Magic 8.3.483, Netgen 1.5.270, KLayout 0.29.1 — inside `efabless/openlane:latest` (`sha256:26719ced90c3…`) |
| EDA (host) | `yowasp-yosys` 0.68 / `yowasp-nextpnr-ecp5` 0.11.1 (WebAssembly) |
| PDK | sky130A via `volare` `c6d73a35f524070e85faff4a6a9eef49553ebc2b` |
| Liberty | `sky130_fd_sc_hd__tt_025C_1v80.lib` (334 usable cells, 94 skipped) |

## Evidence classes

Per `.claude/loop.md` §3.2: `MEASURED`, `SIMULATED`, `FORMALLY_VERIFIED`,
`ANALYTICAL`, `ESTIMATED`, `PROJECTED`, `NOT_IMPLEMENTED`, `BLOCKED`.
Nothing in this repository is `SILICON-MEASURED`, and nothing is
`FPGA-MEASURED`: there is no board.

---

## 1. Claim table

| Claim | Evidence | Status | Risk | Required next step |
|---|---|---|---|---|
| Rail decomposition is bit-exact vs a dense reference | Full test suite green (`pytest -q`, exit 0); 9 new ASIC tests added this session | `SIMULATED` | Low | Keep; extend property/fuzz coverage (loop.md §12) |
| Gemma3 1B full-model rail ≡ dense, 26/26 layers, 0/262144 logit-bit mismatch | `results/gemma_repro.json` from an earlier session; **not re-run here** — `model_data/` weights are not in the working tree | `SIMULATED` (stale) | Medium — headline correctness claim resting on an unreproduced artifact | Re-run `research/reproduce_gemma.py` once weights are present |
| Multiply reduction ≥93% | `railnet.analysis.compute_cost`, analytic over real routing tables | `ANALYTICAL` | Low | None |
| Route-map does not compress (≈1.003× dense) | zlib/RLE/block probes + train-in-rail-basis experiment | `MEASURED` (software) | Low | None — this is a negative result and should stay documented |
| **4-tile grid runs at 83.79 MHz on ECP5** | **NOT REPRODUCED.** `railnet_top_2x2` does not route: `ERROR: IO 's_axis_tvalid' is unconstrained in LPF`. `ecp5_versa.lpf` constrains only clocks, reset, 4 LEDs, PCIe refclk/perst | `BLOCKED` | **High — was published as FPGA-MEASURED** | Either constrain the AXI pins or run out-of-context with `--lpf-allow-unconstrained` and label it an internal-block estimate |
| Per-tile ECP5 P&R timing | `stagea_bram` 209.12 MHz (0 DSP), `stageb_int8` 157.80 MHz, `full_int8_tile` 95.50 MHz — reproduced 2026-09-12 | `MEASURED` (tool estimate, Tier 5; no board) | Low | None |
| Stage-A gather uses 0 DSPs | Reproduced in the same P&R run | `MEASURED` (synthesis) | Low | None — this is RailNet's strongest hardware result |
| **Sky130 synthesis "passed", 1,383 / 327,884 cells** | **FALSE AS STATED.** Those were generic Yosys RTLIL primitives; `abc -liberty` had never run | — | — | Superseded, see next row |
| **Sky130 standard-cell mapping** | `railnet_top`: **39,706 `sky130_fd_sc_hd__*` cells (55 types), 245,247.712 µm² (0.245 mm²)**, 24 `$mem_v2` unmapped. `hardware/asic/gate3_evidence.json` | `MEASURED` (Tier 4, pre-layout) | Low | Gate 5 P&R |
| **Memories cost 9.4× the logic area** | Same design, `memory` vs `memory -nomap`: 317,850 cells / 2,309,721.456 µm² vs 39,706 / 245,247.712. Δ ≈ **2.06 mm²**. `hardware/asic/gate4_memory_area.json` | `MEASURED` | **High — earlier area claims never carried this cost** | Wire the 16b×1024 route memories to `sky130_sram_1kbyte_1rw1r_8x1024_8` pairs |
| SRAM macros "zero LEF/LIB/GDS/SPICE exist" | **No longer true.** sky130A ships complete artifacts for `sky130_sram_1kbyte_1rw1r_32x256_8` and `2kbyte_1rw1r_32x512_8`; the `4kbyte_1rw1r_32x1024_8` the repo models is absent | `OPEN` (shape mismatch, not missing files) | High | See row above |
| `hardware/asic/sky130_sram_macros.v` is integrated | Dead code: no module instantiates it; the RTL infers 24 `$mem_v2` memories instead | `NOT_IMPLEMENTED` | Medium — implies an integration that does not exist | Delete or replace with real macro blackboxes when Gate 4 lands |
| PCIe DMA 720× faster than MMIO | Analytical ratio; no hardware | `ANALYTICAL` | Medium if quoted as measured | Label consistently; needs a board |
| PCIe endpoint works | `MockPCIeBridge` / BFM only; `is_hardware == False` | `SIMULATED` | Medium | Needs a board |
| 200 MHz in sky130 | SDC target only; no STA has ever run | `PROJECTED` | High | Gate 6 — an OpenSTA run at 5 ns is the actual test |
| ReRAM density / energy / TCO | Literature and datasheet assumptions, no PDK or IP | `ESTIMATED` | High — the whole "reprogrammable weight-in-silicon" thesis rests here | A real NVM IP or PDK, or restate as a scenario |
| Gate 5/6/7 (P&R, STA, DRC/LVS) | OpenLane flow launched 2026-09-12 on `railnet_top` at 20 ns; **in progress at time of writing** | `IN PROGRESS` | — | Record the outcome, then repeat at 5 ns |

---

## 2. Defects found and fixed this session

1. **`build_fpga.py` swallowed nextpnr failures.** nextpnr signals failure with
   `SystemExit`; it was caught and ignored, so an unroutable design was recorded
   like a clean run, tagged `FPGA-MEASURED`, and the script exited 0. Status is
   now derived from exit code, `ERROR:` lines and routing completion; a failed
   run publishes no Fmax and the build exits non-zero.
2. **`verify_all_claims.py` fabricated confirmations.** It ran each reproduction,
   kept the exit code, discarded the output, then printed a stored string as
   that run's observation — CLM-05 printed `Execution: FAIL (code 1)` and
   `confirmed via yowasp-nextpnr-ecp5` on consecutive lines. It now prints what
   the run observed and pins `hardware_verified` to `False` throughout.
3. **`benchmark_fpga_prototype.py` could not fail.** `load_routed_fmax()` fell
   back to the target frequency, then to any other target's Fmax, then to a
   hardcoded 83.79 labelled "measured". It now returns `None` unless a target
   actually completed, and the clock carries its provenance.
4. **A hardcoded `"bit_exact_numerical_match": True`** was printed as
   `PASS (100%)` while nothing was ever compared to a hardware output. Removed.
5. **Dry runs overwrote real evidence.** `build_fpga.py --dry-run` wrote to
   `results/fpga_pnr_results.json`, and the test suite runs dry-run builds.
   Dry runs now write elsewhere.
6. **`synth_asic.py` could not run on a clean clone** — its inputs live under
   `hardware/rtl/build/`, which is gitignored. It now regenerates them.

## 3. Structural findings

- **Evidence is gitignored.** Both `build/` and `results/` are excluded, so
  every synthesis artifact the claims cite is untracked. This is the mechanism
  by which the old claims drifted unchallenged. Gate 3 and Gate 4 numbers are
  therefore committed as `hardware/asic/gate3_evidence.json` and
  `gate4_memory_area.json`.
- **`yowasp-yosys` cannot do standard-cell mapping at this scale.** It is
  WebAssembly; Yosys 0.4x there re-parses the 12.8 MB Liberty once per module
  (20+ times observed), and a flattened attempt ran 40+ minutes without
  finishing. Native Yosys in the container does the same work in ~11 s.
- **Docker Desktop will not bind-mount the drive this repo lives on**
  (`mkdir /run/desktop/mnt/host/f: file exists`), so container runs stage inputs
  into a temp directory on C:.

## 4. Honest bottom line

RailNet's software and verification layer is mature and its negative results
(route maps do not compress) are documented rather than buried — that is real
research hygiene. The hardware story is earlier than the documents implied: as
of this audit the project has **one** genuine ASIC-flow result (Gate 3), a
quantified but unbuilt Gate 4, no physical hardware of any kind, and an
FPGA headline figure that does not reproduce. The reprogrammable
weight-in-silicon thesis still rests entirely on ReRAM assumptions with no PDK
or IP behind them.
