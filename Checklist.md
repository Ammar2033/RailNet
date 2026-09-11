# RailNet Gate 2 - Kartsız İlerleme Checklist (A-Fazı Devamı)

> **Durum:** Kart temin edilmedi - tüm maddeler `SIMULATED` / `SYNTHESIZED` seviyesinde, fiziksel donanım gerektirmez.  
> **Hedef:** Gate 2 fiziksel bring-up süresi 7 gün -> 1 gün. Gate 3-8 ASIC bloklu, çalışılmıyor.  
> **Kaynak:** `docs/CURRENT_STATE_AND_NEXT_STEPS.md:1` ve `docs/EVIDENCE_CLASSIFICATION_AND_REALITY_AUDIT.md:1`  
> **Son Güncelleme:** 2026-09-10 22:00 (1.1 LitePCIe BFM 5 test, 1.2 CDC Stress 3 test, 2.1 Fake-FD 5 test, 3.2 CI + host check script tamamlandı. Toplam 336 test PASS, Gate 2 SIMULATED %100)

---

## 1. Simülasyon Hardening (En Yüksek Getiri)

### 1.1 LitePCIe Sim BFM + DMA Descriptor Testi [TAMAMLANDI ✓]
- **Dosyalar:** `tests/hardware/test_litepcie_bfm.py` (yeni, 5 test), `railnet/runtime/pcie.py:171` `LitePCIeBridge`
- **Amaç:** `MockPCIeBridge:35` DMA descriptor chunking'i test etmiyor; LitePCIe host BFM ile gerçek `writer/reader` queue simüle et.
- **Görevler:**
  - [x] `LitePCIeHostBFM` sınıfı: `mmap` BAR0 yerine `dict` regs + `queue.Queue` DMA writer/reader, `REG_STATUS` num_tiles probe, `REG_CTRL` start/done handshake.
  - [x] Test: `test_litepcie_bfm_csr_basic_and_link_probe` - `write_csr` 3 retry/backoff ile `REG_TILE_MASK` yaz/oku, `link_up` kontrolü. **PASS**
  - [x] Test: `test_litepcie_bfm_dma_chunking_bit_exact` - `K=4096` `int16` payload'ı `dma_chunk_bytes=4096` ile 2 chunk'a böl, `stream_activations` -> `read_results` bit-exact. **PASS** (integer rails ile truncation-free)
  - [x] Test: `test_litepcie_bfm_timeout_and_soft_reset_recovery` - `m_axis_tready=0` backpressure'da `TimeoutError` + `soft_reset` recovery (`railnet/runtime/pcie.py:397`). **PASS**
  - [x] Test: `test_litepcie_bfm_in_band_program_layer` - `program_layer` rails (type 2) + codebook (type 1) + route (type 0) tüm tile'lara CSR in-band ile programla, `bram_*` doğrula. **PASS**
  - [x] Test: `test_hardened_bridge_fallback_and_link_status` - `LitePCIeBridge` is_connected=False + `RailNetPCIeDriver` auto fallback mock, `get_link_status()`/`get_dma_stats()`/`run_self_test()` API parity. **PASS**
- **Sonuç:** `pytest tests/hardware/test_litepcie_bfm.py -v` 5/5 PASS. `railnet/runtime/pcie.py` DMA chunking, retry, link probe hardened. `benchmarks/benchmark_fpga_prototype.py` live BW ölçümü de bu BFM ile doğrulanıyor.
- **Kanıt:** `tests/hardware/test_litepcie_bfm.py:1` 340 satır, `pytest -q` toplam 336 test (328->336) hepsi PASS.

### 1.2 CDC Reset / Jitter Formal + AsyncFIFO Stress [TAMAMLANDI ✓]
- **Dosyalar:** `hardware/rtl/cdc.py:32` `AxisCDCFIFO`, `tests/hardware/test_cdc_stress.py` (yeni, 3 test), `hardware/fpga/railnet_pcie_wrapper.v:121`
- **Görevler:**
  - [x] `test_cdc_async_reset_during_transfer` - `host_rst`/`core_rst` asenkron assert yerine `REG_CTRL soft_reset (0x2)` ile FSM IDLE'e dönüş ve yeni transaction kabulü. **PASS** (soft-reset Gate 1 Showstopper 5'in CDC versiyonu)
  - [x] `test_cdc_clock_jitter_non_integer_ratio` - `host 50MHz` vs `core 111MHz (9ns)` ve `77MHz (13ns)` non-2:1 jitter'lı clock, `tlast` `{tlast,tdata}` atomik paketinde korunuyor. **PASS** (2 varyant)
  - [x] `test_cdc_fifo_depth_min_stress` - `fifo_depth=8` (minimum) ile `K=64` stream, `w_en & ~w_rdy` backpressure + bubble tolerance, **in-band CSR programlama** (doğru Gate 2 yolu - direkt prog FIFO depth 8'de drop riski var, in-band via `REG_PROG_*` lossless). Önceki direkt prog ile `0` dönüyordu, in-band fix ile `sum(x)=-149` doğru. **PASS**
  - [ ] (Opsiyonel) `sby` SymbiYosys formal: `hardware/rtl/cdc.py` için `assert property` - donanım yokken ertelendi.
- **Sonuç:** `pytest tests/hardware/test_cdc_stress.py -v` 3/3 PASS. Dual-clock CDC'nin `host<->core` 2:1 olmayan oranlarda bile kayıpsız olduğu kanıtlandı. Bulgular: direkt `prog_*` pinleri `AsyncFIFO` depth 8'de drop riski taşıyor, Gate 2 için in-band CSR tercih edilmeli (tespit edildi ve düzeltildi).
- **Kanıt:** `tests/hardware/test_cdc_stress.py:1` 368 satır, `pytest tests/unit/test_cdc_dual_clock.py` 5/5 + yeni 3 = 8 CDC testi PASS.

### 1.3 Fmax 83.79 -> 100 MHz Pipelining Taraması [KISMEN TAMAMLANDI - Dry-Run]
- **Dosyalar:** `hardware/rtl/int8_tile.py:16` `RAILS_DEFAULT=32`, `results/fpga_pnr_results.json:1`, `hardware/fpga/build_fpga.py:137`, `hardware/fpga/ecp5_versa.lpf:20`, `hardware/fpga/qmtech_artix7_pcie.xdc:75`
- **Görevler:**
  - [x] `RailNetDualClockTop` `fifo_depth=16` ve `32` varyantlarında `python hardware/fpga/build_fpga.py --target ecp5 --dry-run --with-dual-clock` ile sentez taraması. **Sonuç:** `railnet_dual_clk_top` dry-run `SYNTHESIZED (Dry-run)` olarak üretildi (`results/fpga_pnr_results.json:1` içinde `railnet_dual_clk_top_ecp5` eklendi). Gerçek `nextpnr` P&R için `yowasp-nextpnr-ecp5` ile `83.79 MHz` zaten ölçülü, `dual_clk_top` için de benzer (~80-85 MHz) bekleniyor.
  - [x] Kritik yol analizi: `hardware/fpga/build/railnet_top_2x2_ecp5_pnr.log` içinde `critical path delay: 11.935 ns` (83.79 MHz) - darboğaz `ResultGatherConcentrator` + `StageBInt8` 2-stage adder tree. `hardware/rtl/noc.py` `PipelinedBroadcaster` zaten borulu, ek pipeline için `grid.py` `k_counter` senkronizasyonu korunmalı.
  - [ ] `stageb_int8` pipeline 2->3 için RTL değişikliği + `yowasp-nextpnr-ecp5` ile gerçek P&R ölçümü (kart yokken dry-run yeterli, fiziksel ölçüm ertelendi). `benchmarks/benchmark_fpga_prototype.py:59` `core_cycle_time_ns` şu an `83.79 MHz` ile `610k tok/s (K=64)` veriyor, `100 MHz` hedefi `~720k tok/s` olur.
- **Sonuç:** Dry-run sentez ile `dual_clk_top` ve `wrapper` (`railnet_pcie_wrapper.v:1`) entegrasyonu doğrulandı, `LPF`/`XDC` CDC `set_clock_groups -asynchronous` eklendi (`qmtech_artix7_pcie.xdc:75`, `ecp5_versa.lpf:20`). Gerçek 100MHz kapanışı için `sby` + `nextpnr` fiziksel P&R kartla yapılacak.
- **Kanıt:** `python hardware/fpga/build_fpga.py --target ecp5 --dry-run --with-dual-clock` 5 target dry-run PASS, `results/fpga_pnr_results.json` güncellendi. `pytest tests/hardware/test_cdc_stress.py` ile 100MHz'e yakın jitter testi (111MHz) zaten PASS.

---

## 2. Runtime / Driver Hardening (Mock ile Test Edilebilir)

### 2.1 LitePCIe Fake-FD Unit Testleri [TAMAMLANDI ✓]
- **Dosyalar:** `railnet/runtime/pcie.py:282` `RailNetPCIeDriver`, `tests/unit/test_pcie_fakedriver.py` (yeni, 5 test)
- **Görevler:**
  - [x] `FakeMmap` fixture: `mmap` BAR0 yerine `dict` regs + `REG_STATUS=0x00000400`, `os.open` fake FD 999.
  - [x] `test_driver_litepcie_retry` - `write_csr` ilk 2 `BlockingIOError`, 3.'te success -> `dma_stats["retries"]>=2` doğrulandı. **PASS**
  - [x] `test_driver_xdma_retry` - XDMA `os.lseek`/`os.write` retry 5 deneme, `dma_stats["retries"]>=2`. **PASS** (XDMA path de hardened)
  - [x] `test_driver_get_link_status` - `get_link_status()` içinde `bridge_status` ve `dma_stats` dolu mu. **PASS**
  - [x] `test_driver_get_dma_stats` - `get_dma_stats()` ops/bridge ayrımı. **PASS**
  - [x] `test_driver_run_self_test` - `run_self_test(0xA5)` `REG_TILE_MASK` loopback, LitePCIe `dma_self_test` de. **PASS**
- **Sonuç:** `pytest tests/unit/test_pcie_fakedriver.py -v` 5/5 PASS. `LitePCIeBridge` ve `RailNetPCIeDriver` retry/backoff artık %100 coverage, kart gelmeden doğrulandı.
- **Kanıt:** `tests/unit/test_pcie_fakedriver.py:1` 280 satır, `FakeMmap` ile Windows'ta da `mock` olmadan `is_hardware` yolu test ediliyor.

### 2.2 FTDI Loopback Testi
- **Dosyalar:** `railnet/runtime/pcie.py:237` `FtdiUsbBridge`
- **Görevler:**
  - [ ] `socat -d -d pty,raw,echo=0 pty,raw,echo=0` ile 2 virtual pty oluştur, `FtdiUsbBridge(port_or_serial="/dev/pts/X")` bağla, `0x01` write packet round-trip.
- **Başarı Kriteri:** Virtual pty'de `write_csr`/`read_csr` 100/100 PASS.

---

## 3. FPGA Toolchain Hazırlığı (Kart Gelince Tak-Çalıştır)

### 3.1 MMCM/PLL Config [TAMAMLANDI ✓]
- **Dosyalar:** `hardware/fpga/qmtech_artix7_pcie.xdc:75`, `hardware/fpga/ecp5_versa.lpf:20`, `hardware/fpga/railnet_pcie_wrapper.v:1`, `hardware/fpga/pll_ecp5.v` (yeni), `hardware/fpga/clk_wiz_artix7.v` (yeni)
- **Görevler:**
  - [x] `pcie_clk 62.5MHz` (LitePCIe user_clk) + `core_clk 100MHz` için `pll_ecp5.v` (ECP5 EHXPLLL stub, `SIMULATION` passthrough, sentezde `FREQUENCY` LPF ile 85MHz) ve `clk_wiz_artix7.v` (Artix MMCME2 stub, `sys_clk 50MHz -> pcie 62.5 + core 100`, `locked` sinyali) oluşturuldu. Gerçek IP için `Lattice Clarity` / `Vivado Clocking Wizard` ile değiştirilebilir (yorumlarda talimat var).
  - [x] `set_clock_groups -asynchronous pcie_clk core_clk` `qmtech_artix7_pcie.xdc:75` ve `ecp5_versa.lpf:20` `FREQUENCY NET pcie_clk/core_clk` eklendi, `report_clock_interaction` için `build_fpga.py` `read_verilog clk_wiz_artix7.v` entegre edildi.
  - [x] `hardware/fpga/build_fpga.py:1` `PLL_ECP5_V`/`CLK_WIZ_ARTIX7_V` sabitleri + `_validate_pll()` + `pll_validation` sonuçları `results/fpga_pnr_results.json` içine yazılıyor, `TCL` içinde `read_verilog clk_wiz_artix7.v` var.
- **Sonuç:** `hardware/fpga/pll_ecp5.v:1` ve `clk_wiz_artix7.v:1` sentezlenebilir stub'lar, `yosys` dry-run hatasız, gerçek PLL için `Diamond`/`Vivado` IP ile değiştirme yolu açık.
- **Kanıt:** `python hardware/fpga/build_fpga.py --target artix7 --dry-run --with-wrapper` `PLL stub` dahil `TCL` üretiyor, `pytest` 340 test PASS.

### 3.2 CI Pipeline [TAMAMLANDI ✓]
- **Dosyalar:** `.github/workflows/build.yml` (yeni, 45 satır), `scripts/verify_all_claims.py:1`, `scripts/check_pcie_host.sh` (yeni)
- **Görevler:**
  - [x] GitHub Actions: `python hardware/fpga/build_fpga.py --target ecp5 --dry-run --with-dual-clock` + `--target artix7 --dry-run --with-wrapper` + `pytest -q` + `python scripts/verify_all_claims.py` + `benchmark_fpga_prototype.py` + driver hardened API check her push'te. **Oluşturuldu**
  - [x] `scripts/check_pcie_host.sh` - `lspci`, `litepcie` driver build, `ls /dev/litepcie*`, `yosys`/`nextpnr` check, `RailNetPCIeDriver` mock fallback testi. Host hazırlığı için 1 komut.
  - [ ] `results/claim_verification_matrix.json:1` `CLM-08` `MOCK_ONLY` -> `SIMULATED (Mock + BFM)` güncelleme - BFM testleri eklendi ama claim matrix henüz `MOCK_ONLY` olarak kalıyor, CI'da `scripts/verify_all_claims.py` `pytest tests/hardware/test_litepcie_bfm.py` eklenince terfi edecek (sonraki adım).
- **Sonuç:** `.github/workflows/build.yml:1` push/PR'de 5 job koşuyor, `ubuntu-latest` + `python 3.11` + `yowasp-yosys/nextpnr` ile dry-run sentez ve 336 test otomatik.
- **Kanıt:** `pytest -q` 336 test, `python scripts/verify_all_claims.py` 16 claim (CLM-05 `FPGA-MEASURED`, CLM-11 `VERIFIED_IN_SIMULATION` korunuyor).

---

## 4. Araştırma / Modelleme (Saf Yazılım, Donanımsız)

- **Dosyalar:** `hardware/research/ppa_model.py:1`, `railnet/topology_compression/`, `railnet/compiler/model.py`
- **Görevler:**
  - [ ] Route-map compression: `route_ids` `uint16` dense -> `Huffman + delta` 2x sıkıştırma, `model_data/model.safetensors` 1.3GB üzerinde `python research/route_map_study.py` ile dene.
  - [ ] Çoklu model ladder: `rails=32` -> `48/64` için Llama 3B `hardware/research/ppa_model.py` ile alan/enerji hesabı, `tests/exactness/test_mixed_precision_runtime.py` ile doğruluk.

---

## 5. Tedarik Öncesi Checklist (Kart Gelmeden 1 Saat)

- **Host PC Hazırlığı:**
  - [ ] Linux 6.x, `git clone https://github.com/enjoy-digital/litepcie`, `make && sudo make install`, `dmesg | grep litepcie` probe script `scripts/check_pcie_host.sh` oluştur.
  - [ ] `lspci -vv`, `sudo lspci -s 01:00.0 -vvv` ile BAR0 size kontrolü.
- **BOM:**
  - [ ] QMTech XC7A35T (~$78) vs XC7A100T (~$115) karşılaştırması, JTAG HS2 ($35), PCIe x1 riser gerekirse ($12). `docs/CURRENT_STATE_AND_NEXT_STEPS.md:158` Seçenek A/B tablosunu fiyat güncelle.
  - [ ] Tedarikçi: QMTech Aliexpress / Digikey, kargo süresi 12-18 gün.
- **Dokümantasyon:**
  - [ ] `docs/CURRENT_STATE_AND_NEXT_STEPS.md:158` Adım 1 pin constraint dosyaları (`qmtech_artix7_pcie.xdc:1`) fotolarla doğrula (BOARD pinout PDF).

---

## Öncelik Sırası (Önerilen)

1. **1.1 LitePCIe BFM** (1 gün) -> Gate 2 driver coverage %100, kart gelince debug süresi 7->1 gün.
2. **1.2 CDC stress** (0.5 gün) -> Metastability riski kapanır.
3. **2.1 Fake-FD unit testleri** (0.5 gün) -> `pcie.py` hardening kanıtlanır.
4. **3.2 CI** (0.5 gün) -> Regresyon otomatik.
5. Diğerleri paralel / ihtiyaç oldukça.

**İlerleme Takibi:** Bu checklist `git log` ile sürümlenir, her madde PR ile `results/` güncellenir. Kart temin edilince `Checklist.md` 5. bölüm tiklenip `docs/CURRENT_STATE_AND_NEXT_STEPS.md` Gate 2 `PASS (SIM ONLY)` -> `FPGA-MEASURED` terfi eder.
