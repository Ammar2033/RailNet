# RailNet Kritik Sağlamlaştırma Yol Haritası - Kart Gerektirmeyen Detaylı Plan

> **Durum:** Gate 2 A hardening tamamlandı (wrapper dual-clock, pcie.py DMA chunking, build wrapper-aware, 345 test PASS). Gate 3-8 ASIC `BLOCKED`.
> **Amaç:** Demo ölçeği (4 tile, K=64) geçip üretim ölçeği (Gemma 1B, 1152x6912, 96 rail, 2 chiplet) çökmesini kart gelmeden yakalamak.
> **Kapsam:** Bu doküman `Checklist.md:1` içindeki 1.1-3.2 maddeleri değil, onlardan **daha önemli** 4 kritik deliği detaylandırır. 4. madde (route compression) hariç, tamamı `SIMULATED`/`SYNTHESIZED` seviyesinde, fiziksel kart gerektirmez.
> **Tarih:** 2026-09-10
> **Kaynaklar:** `MASTER_ACTION_PLAN.md:1`, `docs/EVIDENCE_CLASSIFICATION_AND_REALITY_AUDIT.md:1`, `results/claim_verification_matrix.json:1`, `hardware/research/ppa_model.py:1`

---

## Yönetici Özeti - Neden Bunlar Daha Önemli

Önceki `Checklist.md` 1.1-3.2 maddeleri **driver/RTL bug fix** seviyesindeydi (retry, CDC, PLL stub). Bu 4 madde **bilimsel geçerlilik ve tapeout fizibilitesi** seviyesinde:

| # | Başlık | Risk Tipi | Kart Gerekir mi | Çökerse Ne Olur |
|---|---|---|---|---|
| **H-A** | Tam Ölçek Tek Katman Doğrulama | Bilimsel | Hayır (CPU + sim) | `exact==total` iddiası 1 katmanda bile çökerse tüm verimlilik tezi geçersiz |
| **H-B** | NoC Ölçeklenebilirlik Kanıtı | Donanım | Hayır (Yosys synth) | 2x2'de 2,350 LUT olan `RailNetTop` 16x16'da (256 tile) kuadratik şişerse 572 mm² hesabı 2 kat sapar, 2 chiplet 4 olur |
| **H-C** | ReRAM Endurance / Retention Modeli | Fizik | Hayır (Python model) | `1e6` write endurance ile 18 model değişiminde ömür biterse "reprogrammable" iddiası Taalas gibi `fixed ROM`'a döner |
| **H-D** | Supply-Chain Reproducibility | Güven | Hayır (Docker) | `yowasp-yosys 0.68` pin'li değilse hakem `repro_full.log` 0 byte olan projeyi reddeder |

---

## H-A: Tam Ölçek Tek Katman Doğrulama (En Kritik)

### Problem
*   `tests/hardware/test_pcie_end_to_end.py:403` `K=8,16,32` sentetik, `tests/unit/test_pcie_fakedriver.py:1` 5 test sentetik, `results/claim_verification_matrix.json` `CLM-11` 10 test hep sentetik. Hiçbir test `model_data/model.safetensors` (1.99GB, Gemma3-1B, 18 katman, `hidden=1152`, `intermediate=6912`, vocab 262k) içindeki **gerçek ağırlık dağılımını** kullanmıyor.
*   `railnet/rails/_repair.py` ve `railnet/compiler/int8.py` `repair_missing_values` / `greedy` routing, sentetik `uniform(-0.5,0.5)` dağılımda %100 kapsama veriyor ama gerçek `bf16` ağırlıklarda `outlier` (large magnitude) ve `NaN`/`inf` kuyruğu var. `MASTER_ACTION_PLAN.md` Faz 3'te `INT8` paletleyici `[-128,127]` aralığında %100 kapsama iddiası, gerçek Gemma `gate_proj` 1152x6912 matrisinde denenmedi.
*   `railnet/runtime/transformer.py` `block_forward` içinde `O(N^2)` `np.concatenate` KVCache fix'i (`TransformerContext`) yapıldı ama `O(1)` circular buffer `K=2048` gerçek sekansla hiç test edilmedi - `research/reproduce_gemma.py` 1B tam modelde OOM vermişti (`df70c6b Fix OOM`).

### Evidence Gap
*   `CLM-11` `VERIFIED_IN_SIMULATION` sadece 4x4 sentetik, `results/final_snapshot.json:1` `evidence_level: SIMULATED` ama `HARDWARE` değil. Hakem sorar: "1152x6912 tek bir `down_proj` `96 rail` ile kayıpsız mı?"
*   `SPEC.md:1` Madde 2 `Weight exactness: float32_to_bf16_bits(reconstructed)==target_bits` - sentetikte `exact==total` ama gerçekte kaç `exact`/`total`?

### Feature Detayı
**Yeni Test:** `tests/exactness/test_full_layer_gemma.py` (yeni, ~250 satır)

```python
# 1. Gerçek modelden tek katman yükle (safetensors, 1152x6912,  ~8M params)
from railnet.safetensors_reader import SafetensorsReader
reader = SafetensorsReader("model_data/model.safetensors")
gate_weight = reader.get_tensor("model.layers.0.mlp.gate_proj.weight")  # bf16 [6912,1152]
# 2. Derle (96 rail, max_terms=4, exact=True, int8=False)
from railnet.compiler.model import compile_model  # veya compile_tensor
from railnet.dtypes import get_dtype
compiled = compile_tensor(gate_weight, dtype="bf16", rails=96, max_terms=4, exact=True)
# 3. CPU sim + cycle-accurate sim ile bit-exact doğrula
from railnet.sim.accelerator import AcceleratorSimulator, HardwareConfig
sim = AcceleratorSimulator(HardwareConfig.edge_asic)  # 64 tile, 500MHz
from railnet.runtime.pcie import MockPCIeBridge
# 4. 1 token forward, dense reference vs rail
logits_rail = rail_forward(x, compiled)  # via rail_linear_fast
logits_dense = dense_forward(x, gate_weight)  # bf16 reference
assert bf16_bits(logits_rail) == bf16_bits(logits_dense)  # SPEC Madde 2
```

*   **Kapsam:** `gate_proj` (1152->6912), `up_proj` (1152->6912), `down_proj` (6912->1152) üçlüsünü ayrı ayrı, sonra `block_forward` bir tam katman olarak. `K=1152` ve `K=6912` iki yön.
*   **Performans:** Tek katman 8M param, `compile_tensor` `coordinate-descent` ile ~2-5 dk, `pytest` içinde `timeout=600s`. OOM olursa `RailNetModel.lean` gibi streaming.

### Dosyalar
*   `railnet/compiler/model.py:1` `compile_model` (mevcut, 96 rail ladder)
*   `railnet/dtypes/bf16.py` `exact_equal`
*   `railnet/sim/accelerator.py:98` `TileSimulator` (stage_a_lanes=3)
*   `research/reproduce_gemma.py` (mevcut, 1B tam model reproduce, ama tek katman izole değil)

### Başarı Kriteri
*   `pytest tests/exactness/test_full_layer_gemma.py -v` 3/3 PASS, her biri `exact==total` raporu `exact/total` logluyor, `repro_full.log` 0 byte değil.
*   `results/claim_verification_matrix.json` `CLM-11` `expected_result` güncellenir: `1152x6912 gate_proj 96 rail exact==total` olarak.

### Evidence Tier
*   `SIMULATED` (CPU + cycle-accurate sim), donanım yok ama **gerçek ağırlıkla** olduğu için `CLM-11` `VERIFIED_IN_SIMULATION` güçlenir.

### Efor / Etki
*   Efor: 1 gün (derleme + OOM fix + bit-exact assert).
*   Etki: **Çok Yüksek** - Tek katmanda bile `exact` düşerse tüm INT8 ladder ve PPA hesabı çöker, tapeout öncesi yakalanır.

---

## H-B: NoC Ölçeklenebilirlik Kanıtı

### Problem
*   `hardware/rtl/grid.py:1` `RailNetGrid` `num_tiles=4` (2x2) için `Yosys synth_xilinx` `8 DSP, 1,441 FF, 2,350 LUT` (`results/rtl_synth.json:1`). `hardware/research/ppa_model.py:1` buradan `169 mm²` (akademik) ve `572 mm²` (ticari) hesaplıyor ama **hiçbir zaman 8x8 (64 tile) veya 16x16 (256 tile) sentezlenmedi.**
*   `hardware/rtl/noc.py:1` `PipelinedBroadcaster` fan-out `W*H` ve `ResultGatherConcentrator` `all_done` ağacı - 4 tile'da `PipelinedBroadcaster` 1 LUT, 256 tile'da `256 * 32` bit broadcast = 8192 wire, `CARRY4` ve `MUXF` sayısı kuadratik şişebilir. `nextpnr` `critical path 11.935 ns` 2x2 için, 16x16'da `>30 ns` (33 MHz) olursa `benchmarks/benchmark_fpga_prototype.py:88` `100 MHz` hedefi çöker.

### Evidence Gap
*   `CLM-05` `83.79 MHz` sadece 2x2 için `FPGA-MEASURED`, `CLM-13` `572 mm² 2 chiplet` sadece `ppa_model.py` analitik, hiç `synth` ile doğrulanmadı. `final_snapshot.json:1` `GATE 5 P&R` `BLOCKED` ama `GATE 3` bile 4 tile için `BLOCKED` - ölçek hiç denenmedi.

### Feature Detayı
**Yeni Script:** `hardware/research/scaling_study.py` (yeni, ~150 satır)

```python
from hardware.rtl.grid import RailNetGrid
from hardware.rtl.top import RailNetTop
from amaranth.back import rtlil
from yowasp_yosys import run_yosys
import json, re

for tiles in [4, 16, 64, 256]:
    grid = RailNetGrid(num_tiles=tiles, rails=32, codebook_depth=64, route_depth=512)
    il = rtlil.convert(grid, name=f"grid_{tiles}")
    # Yosys synth_xilinx -family xc7
    run_yosys(["-p", f"read_rtlil ...; synth_xilinx -family xc7 -top grid_{tiles}; stat"])
    # Parse LUT6, FF, DSP, BRAM from log
    # Plot LUT vs tiles (beklenen lineer: 2,350/4=587 LUT/tile, 256 tile => ~150k LUT)
```

*   **Çıktı:** `results/scaling.json` `{tiles:4: {lut:2350, dsp:8}, tiles:64: {lut:..., dsp:...}, tiles:256: {...}}` ve `scaling.png` (LUT/tile sabit mi?).
*   **Kontrol:** `PipelinedBroadcaster` `out_x_0..255` her biri `32` bit, toplam `8192` bit, `CARRY4` sayısı `tiles * rails` ile lineer mi?

### Dosyalar
*   `hardware/rtl/grid.py:1`, `hardware/rtl/noc.py:1`, `hardware/rtl/top.py:22`
*   `hardware/fpga/build_fpga.py:1` `synth_and_pnr_artix7` (dry-run yeterli, `stat` için `synth_xilinx` yeterli, `nextpnr` gerekmez)
*   `results/rtl_synth.json:1`, `results/fpga_pnr_results.json:1`

### Başarı Kriteri
*   `python hardware/research/scaling_study.py` 4 nokta için `SYNTHESIZED (Xilinx Tech-Mapped)` **PASS**, `LUT/tile` `500-600` aralığında sabit (kuadratik değil), `results/scaling.json` CI'da arşivleniyor.
*   Eğer `256 tile` `LUT > 200k` (Artix-7 33k LUT'u aşıyor) ise `ppa_model.py` 2 chiplet değil 4 chiplet olarak güncellenir.

### Evidence Tier
*   `SYNTHESIZED (Xilinx Tech-Mapped)` (Yosys `stat`), donanım yok ama ölçek kanıtı.

### Efor / Etki
*   Efor: 0.5 gün (Yosys `stat` parse, 4 sentez her biri 30sn).
*   Etki: **Yüksek** - Monolitik vs chiplet kararı bu grafiğe bağlı.

---

## H-C: ReRAM Endurance / Retention Modeli

### Problem
*   `hardware/research/ppa_model.py:1` ve `taalas_vs_railnet_ppa.md` ReRAM'i `1T1R`, `0.0016 µm²`, `0.55 pJ/bit` ideal kabul ediyor, `CLM-02` `FOUNDRY_IP_SPEC_VERIFIED` ama `endurance` ve `retention` hiç yok. Ticari `TSMC 22ULL eReRAM` datasheet: **write endurance ~1e6 cycles, retention 10 yıl @85C, write latency ~100ns, write energy ~10x read**.
*   `RailNetPCIeDriver.program_layer()` her model yüklemede `route 512*4 + cb 64*4 + rail 32*4 = ~2,400` CSR yazıyor (`railnet/runtime/pcie.py:402`). Gemma 1B 18 katman * 3 projeksiyon (qkv, gate, down) = 54 katman * 2,400 = **129,600 write** tek bir `compile_model` yüklemesinde. `1e6 / 129k = 7.7` model değişiminde ömür biter. `MASTER_ACTION_PLAN.md` "saniyeler içinde flash" diyor ama kaç kez?
*   `Gate 2` double-buffering `prog_bank` ile `wear-leveling` var ama hiç modellenmedi.

### Evidence Gap
*   `CLM-02` ve `CLM-15` (`$33.6M` tasarruf) `NRE` hesabı yapıyor ama `OPEX` (ReRAM aşınması) yok. Eğer her müşteri günde 1 model değiştirirse, kart 1 haftada ölürse TCO tersine döner.

### Feature Detayı
**Yeni Dosya:** `hardware/research/reram_endurance.py` (yeni, ~200 satır)

```python
# TSMC 22ULL eReRAM spec (silicon-measured)
ENDURANCE_CYCLES = 1e6
RETENTION_YEARS_AT_85C = 10
WRITE_LATENCY_NS = 100
WRITE_ENERGY_PJ_PER_BIT = 5.5  # 10x read

def model_loads_until_failure(num_tiles=4, rails=32, route_depth=512, layers=54):
    writes_per_layer = num_tiles * (rails + 64 + route_depth)  # ~2,400
    writes_per_model = writes_per_layer * layers
    loads = ENDURANCE_CYCLES / writes_per_model
    return loads  # ~7.7

def retention_with_wear(loads_per_day=1):
    days = ENDURANCE_CYCLES / (writes_per_model * loads_per_day)  # 7.7 gün
    # + retention 10 yıl, hangisi önce?
```

*   **Çıktı:** `results/reram_endurance.json` `{writes_per_model:129600, loads_until_failure:7.7, days_at_1_per_day:7.7, with_wear_leveling_2banks:15.4}`
*   **Çözüm Önerisi:** `prog_bank` ping-pong zaten var, ama `wear-leveling` için `route` ve `cb`'yi de `2x` derinlik yapıp `active_bank` ile round-robin. Veya `ReRAM` yerine `MRAM` (endurance 1e12) trade-off'u `ppa_model.py` içine ekle.

### Dosyalar
*   `hardware/research/ppa_model.py:1`, `railnet/runtime/pcie.py:402` `program_layer`, `hardware/rtl/cdc.py:337` `prog_bank`
*   `results/reram_endurance.json` (yeni), `results/claim_verification_matrix.json` `CLM-02` altına `endurance` notu

### Başarı Kriteri
*   `python hardware/research/reram_endurance.py` **PASS**, `loads_until_failure` < 100 ise `docs/EVIDENCE_CLASSIFICATION_AND_REALITY_AUDIT.md:1` `CLM-02` altına `WARNING: endurance 7.7 loads` eklenir, `Gate 8` `OPEN` kalır.
*   Eğer `wear-leveling` ile `>1000` olursa `MODELLED_AND_VERIFIED` güçlenir.

### Evidence Tier
*   `MODELLED (Datasheet + Analytical)`, donanım yok ama fiziksel limit.

### Efor / Etki
*   Efor: 0.5 gün (datasheet parse, Python model).
*   Etki: **Çok Yüksek** - "Reprogrammable" iddiasının ömrü bu sayıya bağlı.

---

## H-D: Supply-Chain Reproducibility (Docker + Pin)

### Problem
*   `results/fpga_pnr_results.json:1` `achieved_fmax 83.79 MHz` hangi `yowasp-yosys 0.68` commit'i, `amaranth 0.5.5` ve `nextpnr` versiyonu ile üretildi belli değil. `pyproject.toml:1` `amaranth>=0.5` gevşek, `pip freeze` yok. `repro_full.log` 0 byte. Hakem `docker build && pytest` ile aynı `exact==total`'ı alamazsa reddeder.
*   `hardware/fpga/build_fpga.py:1` `yowasp` WASI FS Windows'ta `Can't open log file` hatası veriyor (görüldü), Linux'ta farklı davranıyor.

### Feature Detayı
**Yeni Dosyalar:** `Dockerfile` (yeni, 30 satır), `results/repro_manifest.json` (yeni)

```dockerfile
FROM python:3.11-slim
RUN pip install amaranth==0.5.5 yowasp-yosys==0.68 yowasp-nextpnr-ecp5==0.11.1 numpy==1.26
COPY . /src
WORKDIR /src
RUN pip install -e ".[dev,rtl]"
RUN pytest -q && python hardware/fpga/build_fpga.py --target ecp5 --dry-run --with-dual-clock
```

*   `scripts/verify_all_claims.py` sonuna `pip freeze > results/repro_manifest.json` + `git rev-parse HEAD` ekle.

### Başarı Kriteri
*   `docker build -t railnet:repro . && docker run railnet:repro pytest -q` **PASS**, `results/repro_manifest.json` CI'da arşivleniyor.

### Evidence Tier
*   `SIMULATED` ama `reproducibility` için `THEORETICAL`'den `SIMULATED`'a terfi.

### Efor / Etki
*   Efor: 0.5 gün.
*   Etki: **Orta** - Bilimsel güven için zorunlu.

---

## Öncelik Matrisi

| Sıra | ID | Etki | Efor | Kart Gerekir mi | Bağımlılık |
|---|---|---|---|---|---|
| 1 | H-A Tek Katman Gerçek | Çok Yüksek | 1 gün | Hayır | `safetensors` 2GB RAM gerekir |
| 2 | H-C ReRAM Ömür | Çok Yüksek | 0.5 gün | Hayır | Yok |
| 3 | H-B NoC Ölçek | Yüksek | 0.5 gün | Hayır | `yowasp-yosys` |
| 4 | H-D Docker | Orta | 0.5 gün | Hayır | Yok |

**Önerim:** `H-A` ile başla - 1 gün içinde `exact` düşerse diğerleri anlamsız. `H-A` PASS olursa `H-C` ile "reprogrammable" iddiasının ömrünü kanıtla.

---
*Bu doküman `Checklist.md` 1-3 maddelerinden daha kritik 4 deliği kapatır. Hepsi `SIMULATED`/`SYNTHESIZED` seviyesinde, kart gelmeden yapılabilir. Kart geldiğinde Gate 2 `FPGA-MEASURED` ve Gate 8 `MODELLED_AND_VERIFIED` doğrudan bu dokümandan beslenir.*
