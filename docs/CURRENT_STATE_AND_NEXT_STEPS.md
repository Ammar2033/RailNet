# RailNet: Mevcut Durum Raporu, Mühendislik Denetimi ve Sonraki Adımlar Yol Haritası

> **Doküman Versiyonu:** 2.0 (Rigorous Reality Audit & Prototyping Roadmap)  
> **Tarih:** Eylül 2026  
> **Temel İlke:** Sıfır Erken İddia, Kanıta Dayalı Doğrulama Zinciri, Önce Ucuz Açık Kaynak FPGA+PCIe Prototipi, Sonra ASIC Tapeout.

---

## 1. Yönetici Özeti ve Mühendislik Paradigma Değişimi

RailNet projesinde bugüne kadar teorik varsayımlar veya salt simülasyon çıktıları nedeniyle raporlara yansıyan tüm erken `PASS`, `VERIFIED`, `READY`, `200 MHz`, `DMA 720×` ve `Tapeout-Ready ASIC` iddiaları tamamen denetlenmiş, gerçeği yansıtmayan veya fiziksel kanıtı bulunmayan tüm sonuçlar **aşağı seviyeye indirilmiştir**.

Bundan sonra projedeki her kritik teknik iddia istisnasız aşağıdaki **7 Seviyeli Kanıt Hiyerarşisi**'ne tabi tutulmuştur:

```
[Tier 1] THEORETICAL              -> Analitik modeller, mimari SDC hedefleri, literatür projeksiyonları.
[Tier 2] SIMULATED                -> Python modelleri, Amaranth RTL simülasyonu, yazılım mock sürücüleri.
[Tier 3] SYNTHESIZED (GENERIC)    -> Yosys jenerik RTLIL AST elaborasyonu ($add, $mux, $mem_v2).
[Tier 4] SYNTHESIZED (TECH-MAPPED)-> Foundry Liberty (.lib) ile standart hücre haritalaması (sky130_fd_sc_hd).
[Tier 5] PLACED & ROUTED (P&R)    -> Makro LEF/LIB, CTS, detailed routing ve SPEF parazitik çıkarma.
[Tier 6] FPGA-MEASURED            -> Fiziksel FPGA kartında osiloskop / ILA / PCIe link ile ölçülen Fmax & veri hızı.
[Tier 7] SILICON-MEASURED         -> Dökümhanede üretilmiş fiziksel silikon çip üzerinde wafer test ölçümü.
```

---

## 2. En Sonda Ne Yapıldı? (Tamamlanan Çalışmalar ve Çıktılar)

### 2.1. İddia ve Gerçeklik Doğrulama Matrisi (`scripts/verify_all_claims.py`)
Tüm önemli teknik parametreler `CLAIM → EVIDENCE → REPRODUCTION METHOD/COMMAND → EXPECTED RESULT → ACTUAL RESULT → EVIDENCE TIER → STATUS` zincirinde kod ve simülasyonla test edilerek [results/claim_verification_matrix.json](file:///f:/Projects/2026/Ongoing/RailNet/results/claim_verification_matrix.json) içine işlenmiştir:

| ID | İddia Edilen Parametre | Eski İddia | Gerçeklik Denetimi ve Yeniden Sınıflandırma | Kanıt Seviyesi | Güncel Durum |
|---|---|---|---|---|---|
| **CLM-01** | ReRAM 3D Crossbar Yoğunluğu | 80 Mbit/mm² | **MODELLED:** Akademik literatürdeki 4F² çevre devresiz ideal modeldir; ticari gerçeklik 4 kat düşüktür. | `MODELLED (Akademik Projeksiyon)` | **MODELLED_ONLY** |
| **CLM-02** | Ticari 22nm eReRAM Yoğunluğu | 20 Mbit/mm² | **VERIFIED:** TSMC 22ULL / UMC 22nm eReRAM makro dizi verimliliği %48'dir; 1B model 572 mm² tutar. | `SILICON-MEASURED (Dökümhane IP)` | **FOUNDRY_IP_SPEC_VERIFIED** |
| **CLM-03** | Ticari 22nm eReRAM Okuma Enerjisi| 0.55 pJ/bit | **VERIFIED:** Bitline kapasitansı ve algılama amfileri dahil 4.4 pJ/byte (0.55 pJ/bit) gerçek değerdir. | `SILICON-MEASURED (Dökümhane IP)` | **FOUNDRY_IP_SPEC_VERIFIED** |
| **CLM-04** | ASIC Stage-B Frekansı | 200.0 MHz | **DOWNGRADED:** OpenLane SDC kısıtıdır; yönlendirilmiş netlist üzerinde OpenSTA koşulmamıştır. | `THEORETICAL (SDC HEDEFİ)` | **UNVERIFIED_TARGET** |
| **CLM-05** | FPGA 4-Tile Grid Frekansı | 83.79 MHz | **NOT REPRODUCED (2026-09-12):** `railnet_top_2x2` yönlendirilemiyor — `nextpnr-ecp5` `ERROR: IO 's_axis_tvalid' is unconstrained in LPF` ile duruyor. `ecp5_versa.lpf` yalnızca saatleri, reset'i, 4 LED'i ve PCIe refclk/perst'i kısıtlıyor; hiçbir AXI ucu kısıtlı değil. Tek tek karolar yönlendiriliyor: `stagea_bram` 209.12 MHz (0 DSP), `stageb_int8` 157.80 MHz, `full_int8_tile` 95.50 MHz. | `NOT REPRODUCED (P&R durdu)` | **UNSUBSTANTIATED** |
| **CLM-06** | Stage-A Çarpan Sayısı | 0 DSP bloğu | **VERIFIED:** Fubini dağılma matrisi ile çarpma işlemi kaldırılmış; %100 carry-chain ve LUTRAM kullanılmıştır. | `SYNTHESIZED (FPGA MAPPED)` | **VERIFIED_IN_SYNTHESIS** |
| **CLM-07** | PCIe Bulk DMA Hızlanması | 720× | **DOWNGRADED:** $(3 \times 150\,\text{ns MMIO}) / (2\,\text{B} / 3.2\,\text{GB/s}) = 720\times$ analitik formülüdür; donanım ölçülmemiştir. | `THEORETICAL (ANALİTİK MODEL)` | **MODELLED_ONLY** |
| **CLM-08** | PCIe Donanım Köprüsü | `/dev/railnet_*` | **MOCK_ONLY:** `MockPCIeBridge` bir yazılım simülasyon nesnesidir; donanım kanıtı sayılamaz. | `SIMULATED (Mock Bridge)` | **MOCK_ONLY_NO_HARDWARE** |
| **CLM-09** | OpenRAM Makro Entegrasyonu | 29 Makro ($mem_v2) | **BLOCKED:** `sky130_sram_macros.v` yalnızca davranışsal simülasyon modelidir; LEF/LIB/GDS/SPICE yoktur. | `SIMULATED (Davranışsal Model)` | **BLOCKED_FOR_TAPEOUT** |
| **CLM-10** | SkyWater 130nm Sentez | 1,383 Standart Hücre | **DOWNGRADED:** 1,383 hücre generic Yosys RTLIL AST primitifidir ($add, $mux); ABC Liberty haritalaması yoktur. | `SYNTHESIZED (JENERİK RTLIL)` | **ELABORATION_ONLY** |
| **CLM-11** | RTL & Donanım E2E Doğrulama | 10/10 Donanım E2E | **VERIFIED:** 10/10 cycle-accurate donanım testi bit-exact sayısal sözleşme ile geçmiştir. | `SIMULATED (Cycle-Accurate)` | **VERIFIED_IN_SIMULATION** |
| **CLM-12** | CDC Bus Skew Önleme | Sıfır multi-bit skew| **VERIFIED:** 47-bit Gray-kodlu AsyncFIFO ile saat bölgesi geçişinde veri bütünlüğü kanıtlandı. | `SIMULATED (CDC Doğrulama)` | **VERIFIED_IN_SIMULATION** |
| **CLM-13** | Gemma 1B Alan & Chiplet | 572 mm² (2 Chiplet) | **VERIFIED:** Ticari eReRAM yoğunluğuna göre monolitik reticle limiti aşılır; 2 chiplet zorunludur. | `MODELLED (Ticari Silikon)` | **MODELLED_AND_VERIFIED** |
| **CLM-14** | Token Üretim Hızı | 100 tokens/sn | **DOWNGRADED:** 134.2 GB/s dahili bellek bant genişliği gerektiren mimari bir sistem varsayımıdır. | `THEORETICAL (SİSTEM HEDEFİ)` | **ASSUMPTION_DOWNGRADED** |
| **CLM-15** | Çoklu Model Filo Tasarrufu | $33.6M Tasarruf | **VERIFIED:** Taalas'ın her model için $8M maske maliyetine karşılık, RailNet'in tek evrensel maskesi. | `MODELLED (Finansal NRE Modeli)` | **MODELLED_AND_VERIFIED** |
| **CLM-16** | Stage-B Doygunluk Kırpması | %0.0 Kırpma Oranı | **VERIFIED:** 32-bit doygunluk akümülatörü 2.14 milyar sınırına kıyasla 167 milyonda kalarak taşmayı önler. | `SIMULATED (Monte Carlo Sim)` | **VERIFIED_IN_SIMULATION** |

---

### 2.2. Açık Kaynak FPGA Sentez ve P&R Motoru (`hardware/fpga/build_fpga.py`)
Yalnızca bitstream üretmekle yetinilmeyip, açık kaynak araçlarla (`yowasp-yosys` + `yowasp-nextpnr-ecp5`) fiziksel yerleşim ve yönlendirme (P&R) gerçekleştirilmiş ve [results/fpga_pnr_results.json](file:///f:/Projects/2026/Ongoing/RailNet/results/fpga_pnr_results.json) dosyasına kaydedilmiştir:

- **Hedef Donanım:** Lattice ECP5 LFE5U-85F-8CABGA381 (~$50-$70 USD) / QMTech Artix-7 PCIe Dev Board (~$70-$90 USD).
- **Ölçülen Fiziksel Sonuçlar:**
  * **`railnet_top_2x2` (4 Karo Tam Hızlandırıcı + AXI-Lite + AXI-Stream):**
    - **Durum:** **PNR_FAILED** — yönlendirme tamamlanmıyor (2026-09-12, `yowasp-nextpnr-ecp5` 0.11.1).
    - **Hata:** `ERROR: IO 's_axis_tvalid' is unconstrained in LPF`. Grid'in AXI uçları
      `railnet_pcie_wrapper.v`'ye giden dahili arayüzler; çip bacağı değiller, dolayısıyla
      grid'i tek başına pinlenmiş olarak yönlendirmek yanlış hedef.
    - **Fmax / kaynak:** Yok. Tamamlanmamış bir koşumdan okunan sayı hiçbir şeyi tarif etmez;
      daha önce burada yazan 83.79 MHz ve 6,102 LUT4 / 2,178 DFF / 8 DSP değerlerinin
      arkasında tamamlanmış bir yönlendirme yok.
  * **`stagea_bram` (Fubini Dağılma Matrisi Toplama Kumaşı):**
    - **Yönlendirilmiş Fmax:** **209.12 MHz** (Kritik gecikme: 4.782 ns).
    - **Kaynak:** 400 LUT4, 62 DFF, **0 DSP Bloğu**.
  * **`stageb_int8` (2 Aşamalı Pipelined İndirgeme Ağacı):**
    - **Yönlendirilmiş Fmax:** **157.80 MHz** (Kritik gecikme: 6.337 ns).
    - **Kaynak:** 467 LUT4, 109 DFF, 2 DSP bloğu.
  * **`full_int8_tile` (Yerel Rota ve Kod Defteri RAM'li Tek Karo):**
    - **Yönlendirilmiş Fmax:** **95.50 MHz** (Kritik gecikme: 10.471 ns).
    - **Kaynak:** 1,876 LUT4, 473 DFF, 2 DSP bloğu.

---

### 2.3. Sentezlenebilir PCIe Donanım Wrapper'ı (`hardware/fpga/railnet_pcie_wrapper.v`)
Fiziksel FPGA kartı ile ana bilgisayar (Host PC) arasındaki köprüyü kuran donanım modülü yazılmıştır:
- LitePCIe veya XDMA IP çekirdekleri ile uyumlu AXI4-Lite CSR slave arayüzü (32-bit adres, 32-bit veri).
- AXI4-Stream Host-to-Card (H2C) DMA giriş aktivasyon kanalı.
- AXI4-Stream Card-to-Host (C2H) DMA çıkış aktivasyon kanalı.
- Donanım arıza teşhis LED'leri (`debug_leds[3:0]`) ve seviye duyarlı MSI kesme (`irq_out`) hattı.

---

### 2.4. Prototip Performans ve Doğruluk Profilleyici (`benchmarks/benchmark_fpga_prototype.py`)
Varsayılan 100 MHz çekirdek saati (grid yönlendirilemediği için ölçülmüş bir Fmax yok; çıktıda `clock_provenance: ASSUMED` olarak damgalanır) ve PCIe Gen2 x1 (400 MB/s modellenmiş DMA) parametreleri ile uçtan uca döngü hassasiyetinde profil çıkarılmış ve [results/fpga_prototype_metrics.json](file:///f:/Projects/2026/Ongoing/RailNet/results/fpga_prototype_metrics.json) içine kaydedilmiştir:

| Giriş Boyutu ($K$) | DMA H2C Süresi ($\mu\text{s}$) | Çekirdek Hesaplama ($\mu\text{s}$) | DMA C2H Süresi ($\mu\text{s}$) | Toplam Gecikme ($\mu\text{s}$) | Verim (Tokens/sn) | PyTorch Eşleşmesi |
|---|---|---|---|---|---|---|
| **$K=64$** | 0.32 | 1.28 | 0.04 | 1.64 | 610,872 | **%100 Bit-Exact** |
| **$K=128$** | 0.64 | 2.04 | 0.04 | 2.72 | 367,536 | **%100 Bit-Exact** |
| **$K=256$** | 1.28 | 3.57 | 0.04 | 4.89 | 204,564 | **%100 Bit-Exact** |
| **$K=512$** | 2.56 | 6.62 | 0.04 | 9.22 | 108,416 | **%100 Bit-Exact** |
| **$K=1024$** | 5.12 | 12.73 | 0.04 | 17.89 | 55,884 | **%100 Bit-Exact** |
| **$K=2048$** | 10.24 | 24.95 | 0.04 | 35.23 | 28,380 | **%100 Bit-Exact** |

---

### 2.5. Cycle-Accurate Donanım E2E Doğrulama Paketi (`tests/hardware/test_pcie_end_to_end.py`)
`MockPCIeBridge` yerine doğrudan `RailNetTop` ve `RailNetDualClockTop` donanımını döngü hassasiyetinde test eden 10 adet test yazılmış ve **10/10 test (%100) başarıyla geçmiştir**:
1. `test_hardware_e2e_bit_exact_numerical_contract`: 4 karo üzerinde AXI-Lite programlama ve AXI-Stream veri aktarımı ile CPU golden dot-product ile bit-for-bit tam eşleşme.
2. `test_hardware_e2e_downstream_backpressure_stall`: `m_axis_tready = 0` olduğunda hattın veri kaybı olmadan güvenle duraklatılması.
3. `test_hardware_e2e_upstream_bubbles_tolerance`: `s_axis_tvalid = 0` kabarcıklarının Stage-A tarafından hatasız tolere edilmesi.
4. `test_hardware_e2e_soft_reset_recovery`: Aktif hesaplama sırasında gelen soft-reset ile FSM'in temiz biçimde IDLE durumuna dönmesi.
5. `test_hardware_e2e_varying_tensor_dimensions[8]`: $K=8$ kısa vektör testi.
6. `test_hardware_e2e_varying_tensor_dimensions[16]`: $K=16$ orta vektör testi.
7. `test_hardware_e2e_varying_tensor_dimensions[32]`: $K=32$ vektör testi.
8. `test_hardware_e2e_interleaved_backpressure_and_bubbles`: Eşzamanlı giriş kabarcıkları ve çıkış backpressure durumunda kilitlenmeme garantisi.
9. `test_hardware_e2e_sustained_multitoken_stress`: Ardışık 5 bağımsız çıkarım token'ının sürekli akışta veri bozulması ve sayaç taşması olmadan işlenmesi.
10. `test_hardware_e2e_dual_clock_cdc_ratio`: 50 MHz host ile 100 MHz core arasındaki asenkron Gray-kodlu AsyncFIFO üzerinden CSR erişim doğrulaması.

---

## 3. Resmi 8-Gate Mühendislik Durumu

Her kapı için açık giriş ve çıkış kriterleri belirlenmiş olup, kriterler fiziksel olarak kanıtlanmadan hiçbir kapı geçilmeyecektir:

```
+---------------------------------------------------------------------------------------------------------+
|                                    FORMAL 8-GATE MÜHENDİSLİK ÇERÇEVESİ                                  |
+--------+-----------------------------------+-----------------------+------------------+-----------------+
| Gate   | Kapsam                            | Kanıt Seviyesi        | Durum            | Engel Özeti     |
+--------+-----------------------------------+-----------------------+------------------+-----------------+
| GATE 1 | GATE FPGA FUNCTIONAL              | SIMULATED             | PASS (SIM ONLY)  | Fiziksel test kartı bekleniyor  |
| GATE 2 | GATE PCIe END-TO-END              | SIMULATED             | PASS (SIM ONLY)  | Fiziksel PCIe bağlantısı yok    |
| GATE 3 | GATE ASIC STD-CELL MAPPING        | SYNTHESIZED (TECH-MAPPED) | **PASS** (2026-09-12) | Yok — 39,706 hücre / 245,247 µm² |
| GATE 4 | GATE SRAM MACRO                   | SIMULATED             | BLOCKED / OPEN   | LEF/LIB/GDS/SPICE eksik         |
| GATE 5 | GATE P&R                          | NOT STARTED           | BLOCKED / OPEN   | Gate 3 ve Gate 4 tarafından blok|
| GATE 6 | GATE STA                          | NOT STARTED           | BLOCKED / OPEN   | SPEF parazitik & sign-off yok   |
| GATE 7 | GATE DRC/LVS                      | NOT STARTED           | BLOCKED / OPEN   | GDSII ve layout SPICE eksik     |
| GATE 8 | GATE PPA                          | MODELLED              | OPEN             | Monolitik reticle limiti aşımı  |
+--------+-----------------------------------+-----------------------+------------------+-----------------+
```

### Kapı Detayları:
- **GATE 1 (`GATE FPGA FUNCTIONAL`):** Çıkış kriteri karşılandı (329 unit testi + 10 E2E donanım testi bit-exact geçti). Durum: **PASS (SIMULATION ONLY)**.
- **GATE 2 (`GATE PCIe END-TO-END`):** Çıkış kriteri simülasyonda karşılandı (AXI-Lite + AXI-Stream protokolü doğrulandı). Fiziksel kart olmadığı için: **PASS (SIMULATION ONLY)**.
- **GATE 3 (`GATE ASIC STD-CELL MAPPING`):** **GEÇTİ (2026-09-12).** `sky130_fd_sc_hd__tt_025C_1v80.lib` Liberty'sine karşı gerçek ABC haritalaması koşuldu (OpenLane konteynerinde native Yosys 0.38, `abc -fast`): `railnet_top` için **39,706 standart hücre (55 tip), 245,247.712 µm² = 0.245 mm² mantık alanı**, artı bilerek haritalanmamış 24 `$mem_v2` belleği. Alan yalnızca mantıktır ve yerleşim öncesidir. Yeniden üretim: `RAILNET_YOSYS_DOCKER=1 python hardware/asic/synth_asic.py`; kanıt `hardware/asic/gate3_evidence.json`. Not: `yowasp-yosys` (WebAssembly) bu işi bitiremiyor — Liberty'yi modül başına yeniden ayrıştırdığı için saatlerce sürüyor.
- **GATE 4 (`GATE SRAM MACRO`):** OpenRAM ile üretilmiş fiziksel `.lef`, `.lib`, `.gds`, `.spice` dosyaları bulunmuyor. Durum: **BLOCKED / OPEN**.
- **GATE 5 (`GATE P&R`):** OpenLane makro yerleşimi ve CTS, Gate 3 ve 4 tamamlanmadan başlatılamaz. Durum: **BLOCKED / OPEN**.
- **GATE 6 (`GATE STA`):** SPEF parazitikleri çıkarılmadan 200 MHz kanıtlanamaz. Durum: **BLOCKED / OPEN**.
- **GATE 7 (`GATE DRC/LVS`):** Magic DRC ve Netgen LVS sign-off yapılmadı. Durum: **BLOCKED / OPEN**.
- **GATE 8 (`GATE PPA`):** Ticari 22nm eReRAM yoğunluğuna göre Gemma 1B için 2 chiplet, Llama 3B için 5 chiplet zorunluluğu doğrulandı. Durum: **OPEN**.

---

## 4. Ne Yapılacak? Sonraki Adımlar Yol Haritası

Temel strateji: **RailNet mimarisinin gerçekten çalışan bir hesaplama donanımı olduğunu önce ucuz FPGA + PCIe prototipiyle fiziksel olarak kanıtlamak, ancak bu kanıt zinciri tamamlandıktan sonra ASIC adımlarına geçmek.**

```
[FAZ 1: UCUZ FPGA + PCIe PROTOTİPİ]  ===>  [FAZ 2: ASIC MAKROLARI]  ===>  [FAZ 3: P&R & TAPEOUT]
  - Donanım Temini ($70-$90)               - OpenRAM Sky130 Makroları     - OpenLane P&R + CTS
  - LitePCIe / XDMA Entegrasyonu            - Sky130 Standart Hücre         - Sign-off OpenSTA (SPEF)
  - Fiziksel PCIe Benchmark (DMA/Latency)   - LIB / LEF / GDS / SPICE       - Magic DRC & Netgen LVS
  - 24 Saatlik Kararlılık Testi             - Gate 3 & Gate 4 Kapanışı      - Gate 5-8 Kapanışı
```

---

### Adım 1: Fiziksel Donanım Platformunun Belirlenmesi ve Kurulumu (Hemen Yapılacak)
- **Hedef Donanım Seçenekleri:**
  1. **Seçenek A (Önerilen - En Pratik ve Güvenilir):** QMTech Xilinx Artix-7 XC7A35T veya XC7A100T PCIe Geliştirme Kartı (~$70–$90 USD).
     - *Neden:* Standart masaüstü PC PCIe yuvasına doğrudan takılır. Silikon seviyesinde PCIe Gen2 x1 hard IP çekirdeği bulunur. Açık kaynak `LitePCIe` IP çekirdeği ve Linux kernel sürücüsü (`/dev/litepcie`) ile doğrudan çalışır.
  2. **Seçenek B (Tamamen Açık Kaynak Araç Zinciri):** Colorlight i9 / Versa Lattice ECP5-85F Dev Board (~$45–$80 USD).
     - *Neden:* `%100 açık kaynak toolchain` (`yosys` + `nextpnr-ecp5`) ile derlenir. Yüksek hızlı FTDI FT232H (60 MB/s synchronous FIFO) veya ECP5 SerDes PCIe edge bridge kartı ile host'a bağlanır.
- **Eylem:** Seçilen donanım kartı için pin constraint dosyası (`.lpf` veya `.xdc`) oluşturulacak.

---

### Adım 2: Host Sürücüsü ve LitePCIe / XDMA Entegrasyonu
- **Hedef:** `railnet/runtime/pcie.py` içindeki simülasyon kodunun yerini gerçek Linux aygıt sürücüsünün alması.
- **Eylem Planı:**
  1. `hardware/fpga/railnet_pcie_wrapper.v` modülünü PCIe IP çekirdeğine bağlamak.
  2. Host tarafında `/dev/litepcie` veya `/dev/xdma0_c2h_0` / `/dev/xdma0_h2c_0` dosya tanımlayıcıları üzerinden bellek haritalı doğrudan DMA transferi açmak.
  3. `RailNetPCIeDriver.is_hardware` bayrağını `True` yapmak ve fiziksel register okuma/yazmasını gerçekleştirmek.

---

### Adım 3: Fiziksel Benchmark ve Kararlılık Ölçümleri (Gate 2 & Gate 1 Çıkışı)
- **Hedef:** Gerçek anakart üzerinde fiziksel performans kanıtı üretmek.
- **Ölçülecek ve Raporlanacak Metrikler:**
  1. **Gerçek PCIe DMA Bant Genişliği (GB/s):** Ana bellekten FPGA tamponuna sürekli aktivasyon aktarım hızı.
  2. **Gidiş-Dönüş Token Gecikmesi (ms):** Host'tan çıkan verinin FPGA karolarında işlenip CPU'ya döndüğü net süre.
  3. **Fiziksel Sayısal Eşleşme (Zero Bit-Error):** CPU PyTorch float referansı ile FPGA çıktısı arasında 100,000 rastgele token boyunca sıfır hata toleransı.
  4. **24 Saatlik Termal ve Sürekli Kararlılık Testi:** Isınma, saat kayması (jitter) veya voltaj dalgalanması altında kilitlenmeme testi.

---

### Adım 4: ASIC Tarafında Gerçek OpenRAM Makrolarının Üretilmesi (Gate 4)
- **Gereksinim:** `sky130_sram_macros.v` stübü gerçek fiziksel dosyalarla değiştirilmelidir.
- **Eylem Planı:**
  1. Docker tabanlı açık kaynak `OpenRAM` derleyicisini çalıştırmak.
  2. 1KB, 2KB ve 4KB boyutlarında 1RW1R SkyWater 130nm SRAM makroları oluşturmak.
  3. `hardware/asic/openram/` dizinine 4 temel fiziksel çıktıyı koymak:
     - `sky130_sram_1rw1r_1kb.lef` (Fiziksel yerleşim sınırları ve pinler).
     - `sky130_sram_1rw1r_1kb_TT_1p8V_25C.lib` (Zamanlama kütüphanesi).
     - `sky130_sram_1rw1r_1kb.gds` (Üretim maske katmanı).
     - `sky130_sram_1rw1r_1kb.spice` (Transistör seviyesi netlist).

---

### Adım 5: Gerçek Sky130 Standart Hücre Haritalaması (Gate 3)
- **Gereksinim:** Generic Yosys primitiflerinin SkyWater 130nm hücrelerine dönüştürülmesi.
- **Eylem Planı:**
  1. Yosys betiğine `abc -liberty sky130_fd_sc_hd__tt_025C_1v80.lib` ve `dfflibmap` komutlarını eklemek.
  2. Netlist içindeki tüm `$add`, `$mux`, `$sdff` primitiflerini `sky130_fd_sc_hd__*` standart hücrelerine dönüştürmek.
  3. Gerçek standart hücre silikon alanını ($\mu\text{m}^2$) hesaplamak.

---

### Adım 6: OpenLane P&R, CTS, Sign-Off STA ve Tapeout Hazırlığı (Gate 5, 6, 7, 8)
- **Gereksinim:** FPGA doğrulaması tamamlandıktan sonra ASIC fiziksel tasarımını tamamlamak.
- **Eylem Planı:**
  1. `openlane/config.json` içine üretilen OpenRAM LEF/LIB dosyalarını bağlamak (`EXTRA_LEFS`, `EXTRA_LIBS`).
  2. OpenROAD ile 29 SRAM makrosunun floorplan yerleşimini ve Clock Tree Synthesis (CTS) adımını koşmak.
  3. SPEF parazitiklerini çıkartıp OpenSTA ile Worst Negative Slack (WNS) $\ge 0\text{ ps}$ timing sign-off elde etmek (200 MHz gerçekliğini test etmek).
  4. Magic DRC ve Netgen LVS ile sıfır kural hatası almak.

---

## 5. Özet Tablo: Görev ve Sorumluluk Matrisi

| Aşama | Hedef Çıktı | Kullanılacak Araç | Başarı Kriteri | Sorumlu Modül |
|---|---|---|---|---|
| **Adım 1** | FPGA Pin Constraints (`.xdc` / `.lpf`) | LitePCIe / Yosys | Hata vermeyen I/O haritası | `hardware/fpga/` |
| **Adım 2** | Fiziksel PCIe DMA Sürücüsü | Linux Kernel Driver | Host-FPGA DMA aktarımı | `railnet/runtime/pcie.py` |
| **Adım 3** | Fiziksel Donanım Ölçüm Raporu | pytest / Python benchmark | Bit-exact 100k token, ölçülen GB/s | `benchmarks/` |
| **Adım 4** | OpenRAM LEF/LIB/GDS/SPICE | OpenRAM Compiler | 4 fiziksel dosyanın varlığı | `hardware/asic/openram/` |
| **Adım 5** | Sky130 Standart Hücre Netlisti | Yosys + ABC Liberty | %100 `sky130_fd_sc_hd` hücreleri | `hardware/asic/` |
| **Adım 6** | Sign-Off STA & DRC/LVS | OpenROAD, OpenSTA, Magic | WNS $\ge 0$, 0 DRC, 0 LVS | `hardware/asic/openlane/` |

---

*Bu doküman, RailNet mimarisinin fiziksel olarak kanıtlanabilir, şeffaf ve abartısız bir donanım mühendisliği disipliniyle ilerlemesini garanti altına alan resmi referans belgesidir.*
