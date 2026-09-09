# ROADMAP — RailNet

> **RailNet: Reprogrammable Weight-in-Silicon AI Inference Architecture**  
> Bu yol haritası, donanım RTL sentezi, PPA analizi ve çoklu model ailesi desteği ile güncellenmiş aktif geliştirme durumunu ve 4 aşamalı eylem planını içerir.  
> 📌 *Kapsamlı teknik analiz ve Taalas karşılaştırması için [MASTER_ACTION_PLAN.md](MASTER_ACTION_PLAN.md) belgesini inceleyin.*

---

## 🚀 Aktif 4 Aşamalı Eylem Planı (Active Milestone Plan)

| Adım | Başlık | Durum | Ana Odak ve Çıktılar |
| :---: | :--- | :---: | :--- |
| **Adım 1** | **Çoklu Model Ailesi Desteği** | **TAMAMLANDI ✅** | Llama-3/3.2, Qwen-2.5, Gemma-3; SwiGLU, Meta Llama-3 RoPE scaling, $O(1)$ önceden tahsisli `KVCache`, sentetik E2E testler (229 test geçiyor). |
| **Adım 2** | **Hızlı C++/SIMD CPU Çekirdeği** | **TAMAMLANDI ✅** | AVX2/FMA + OpenMP native C++ DLL + Numba parallel JIT fallback; NumPy'a göre **5.0x - 7.5x**, Numba'ya göre **1.4x - 1.9x** hızlanma. Otomatik terfi zinciri ve tam E2E testleri. |
| **Adım 3** | **Çoklu Dtype Desteği (INT8/FP16)** | **TAMAMLANDI ✅** | INT8 tamsayı ray paleti ve derleme merdiveni, karma hassasiyetli (mixed-precision) yönlendirme, 179 MHz sıfır-DSP integer Stage-A boru hattı, %86.9 LUT tasarrufu, 286 test. |
| **Adım 4** | **Sanal Donanım / Saykıl Simülatörü** | **TAMAMLANDI ✅** | Cycle-accurate hızlandırıcı simülatörü (`railnet/sim/accelerator.py`), BRAM RMW/hazard/NoC mesh broadcast/PCIe DMA modellemesi, `RailNetDevice.sim()` ve telemetri CLI. Toplam 296 test. |

---

## 📊 Faz Bazlı Geliştirme Durumu (Phase Status)

### Faz 1: Araştırma ve Konsept Kanıtlama (Research Proof) ✅
* Paylaşımlı ray (shared rails) ve bit-pattern yönlendirme tezi kanıtlandı.
* Gerçek Gemma BF16 ağırlıkları üzerinde $\ge 93.31\%$ çarpma tasarrufu ölçüldü.

### Faz 2: Tam Yazılım Yürütme ve Kayıpsızlık (Gemma3 1B) ✅
* 182 / 182 lineer tensör kayıpsız (lossless) şekilde derlendi (`railnet compile --resume`).
* 26 / 26 katman gizli durumları (hidden states) ve 262,144 logit biti referans dense model ile **birebir bit-exact** eşleşti (`results/gemma_repro.json`).

### Faz 3: Deterministik Çıkarım ve Üretim (Deterministic Generation) ✅
* Greedy decode ve tokenizasyon entegrasyonu tamamlandı.
* Bellek haritalı (mmap) satır okuma ile embedding ve tied LM-head materialization'sız çalıştırıldı.

### Faz 4: Çoklu Model Ailesi Desteği (Multi-Model Families) ✅
* **Llama-3 / 3.2 Desteği:** `LlamaAdapter` (1B, 3B), standart $w$ RMSNorm, SwiGLU ($\text{silu}(\text{gate}) \cdot \text{up} \to \text{down}$), Meta Llama-3 RoPE scaling (`factor=32.0`, `high/low freq factors`).
* **Qwen-2 / 2.5 Desteği:** `QwenAdapter` (0.5B, 1.5B), QKV attention bias vektörleri, $\theta=1000000.0$ RoPE frekansı.
* **Polimorfik Context:** `TransformerContext` tabanlı dinamik mimari dağıtımı (`create_context()`).
* **$O(1)$ Önceden Tahsisli KV-Cache:** `KVCache` sınıfı ile dairesel tampon tahsisi; her token adımında `np.concatenate` bellek kopyalama maliyeti sıfırlandı.
* **Sentetik Model Doğrulaması:** Disk üzerinde hafızada sentetik safetensors modelleri oluşturularak derleme ve rail-vs-dense kosinüs benzerliği ($>0.999$) test edildi (`tests/unit/test_transformer_multimodel.py`).

### Faz 5: Donanım RTL Mimarisi ve Mantık Sentezi (RTL Prototype) ✅
* **Saf Amaranth 0.5 HDL Kütüphanesi ([`hardware/rtl/fp32.py`](hardware/rtl/fp32.py)):** IEEE-754 uyumlu dönüştürücüler (`Bf16ToFp32`, `Fp32ToBf16`), toplayıcılar (`CombFp32Adder`, `PipelinedFp32Adder`) ve çarpıcı (`Fp32Multiplier`).
* **3-Şeritli Hızlandırıcı Karosu ([`hardware/rtl/bf16_tile.py`](hardware/rtl/bf16_tile.py)):** `RouteDecoder`, sıfır duraklamalı bypass ağlı `StageABf16Bram` ve FP32 akümülatörlü `StageBBf16`.
* **Xilinx Sentez Kanıtı (Yosys):** Stage-A gather mantığı **0 DSP** (725 LUT, 105 FF); 3-şeritli tam karo **yalnızca 2 DSP** tüketti (`results/rtl_synth.json`).
* **ECP5 Fiziksel Yerleşim ve Zamanlama (P&R):**
  - Tamsayı Stage-A: **178.95 MHz** (5.58 ns).
  - BF16 Stage-A: **34.49 MHz** (40nm FPGA üzerinde; 16nm/22nm ASIC'te **400–600 MHz** projeksiyonu).

### Faz 6: ASIC PPA ve Taalas TCO Analizi (ASIC Feasibility) ✅
* [`hardware/research/ppa_model.py`](hardware/research/ppa_model.py) geliştirildi.
* **Zar Alanı:** Gemma3 1B RailNet ReRAM (Path B) ile tek bir **$169.3 \text{ mm}^2$ monolitik zar** içine sığıyor (Taalas $125.4 \text{ mm}^2$). SRAM ise $4,330 \text{ mm}^2$ (13 chiplet) gerektirdiğinden imkansız.
* **Filo TCO Üstünlüğü:** 10.000 çip ve 5 model senaryosunda Taalas $43.5M maliyet çıkarırken; RailNet yeniden programlanabilir tek bir evrensel maske setiyle **$9.6M ($955/çip)** maliyet sunar (**$33.9M net tasarruf, %78 indirim**).
* Sonuçlar [`hardware/research/taalas_vs_railnet_ppa.md`](hardware/research/taalas_vs_railnet_ppa.md) raporunda belgelendi.

---

## 🎯 Yaklaşan Fazlar (Upcoming Phases)

### Faz 7: Yüksek Hızlı C++ CPU Çekirdeği (Optimized CPU Runtime) ✅ *(Adım 2 - TAMAMLANDI)*
- AVX2 / FMA intrinsics ile C++ SIMD gather & accumulate motoru (`railnet/csrc/rail_kernel.cpp`).
- Çıkış nöronları üzerinde OpenMP çoklu iş parçacığı paralelleştirmesi (12 çekirdek).
- Derleyici/ikili bulunmayan durumlar için Numba `@njit(parallel=True)` fallback mekanizması.
- Dinamik dispatch (`CPP > Numba > NumPy`) ve hem FP32 hem FP64 hassasiyet desteği.
- Sonuç: NumPy referansına göre **5.0x - 7.5x**, Numba'ya göre **1.4x - 1.9x** net hızlanma.

### Faz 8: Çoklu Dtype ve INT8 Kuantizasyon Merdiveni ✅ *(Adım 3 - TAMAMLANDI)*
- INT8 tamsayı ray paletleyicisi ve simetrik kuantizasyon (`railnet/compiler/int8.py`).
- 24-32 elemanlı kanonik integer tabanıyla tüm INT8 değerlerinin kayıpsız (0 hata) 2-3 terimle deterministik temsili.
- Karma hassasiyetli model derleme (`compile_model(..., mixed_precision=True)`): Attention BF16, MLP INT8.
- C++ AVX2 SIMD ve Numba INT8 çekirdekleri (`railnet_linear_int8_w8a_float`, `railnet_linear_int8_w8a16`).
- Dedicated Amaranth INT8 RTL karosu (`hardware/rtl/int8_tile.py`): Yosys sentezinde **%86.9 LUT azalması** (3,711 -> 486 LUT) ve $500 çip maliyeti kanıtı.
- Toplam 286 test ile %100 doğrulandı.

### Faz 9: Sanal Donanım ve Saykıl Hassasiyetinde Simülatör ✅ *(Adım 4 - TAMAMLANDI)*
- `railnet/sim/accelerator.py` ile BRAM RMW, forward bypass hazard duraklamaları ve Stage-B akümülatör gecikmelerinin saykıl saykıl modellenmesi.
- 2D NoC mesh broadcast ve PCIe Gen4 DMA transfer gecikmelerinin hesabı.
- `RailNetDevice.sim(profile=...)` şeffaf çalışma zamanı aygıtı ve CLI aracı (`python -m railnet.sim.cli`).
- Sayısal logit denklik garantisi ve kapsamlı telemetri raporu (Sub-watt < 1W doğrulaması, 66,000+ Tokens/Joule). Toplam 296 testle doğrulandı.

### Faz 10: Çoklu-Karo NoC ve AXI4-Stream/Lite Donanım Üst Düzeyi ✅ *(v0.8 - TAMAMLANDI)*
- `hardware/rtl/top.py` ile parametrik 2D Grid, AXI4-Lite CSR (`AxiLiteCsr`) ve AXI4-Stream DMA giriş/çıkış arayüzü.
- `PipelinedBroadcaster` ve `ResultGatherConcentrator` ile düşük tel yüklü NoC ağı.
- `hardware/rtl/export.py` ile 7,149 satırlık sentezlenebilir Verilog-2001 (`railnet_top.v`) dışa aktarımı.
- Yosys sentezinde 4-karolu komple çip tasarımı: **8 DSP, 1,441 FF, 2,350 LUT** (tek bir BF16 karosundan bile daha küçük!).
- Toplam 299 testle %100 doğrulandı.

### Faz 11: PCIe/FPGA Donanım Sürücüsü ve Açık Kaynak ASIC Tapeout Hazırlığı (SkyWater 130nm) ✅ *(v0.9 - TAMAMLANDI)*
- `railnet/runtime/pcie.py`: Linux Xilinx XDMA/QDMA PCIe endpoints (`/dev/xdma*`) ve `MockPCIeBridge` sıfır-kopyalama donanım sürücüsü.
- `railnet/runtime/engine.py` ve `railnet/runtime/transformer.py`: `RailNetDevice.pcie()` üzerinden uçtan uca canlı model çıkarım entegrasyonu.
- `hardware/asic/wishbone_to_axi.v`: Caravel SoC RISC-V Wishbone B4 veriyolunu AXI4-Lite CSR'a dönüştüren donanım köprüsü.
- `hardware/asic/caravel_railnet.v`: Efabless Caravel SoC kullanıcı projesi makro sarmalayıcısı (güç pinleri, Logic Analyzer, GPIO telemetri).
- `hardware/asic/openlane/config.json`: SkyWater 130nm (`sky130_fd_sc_hd`) 50 MHz P&R akış konfigürasyonu.
- `hardware/asic/synth_asic.py`: Yosys ASIC sentez doğrulaması (200,178 mantık kapısı, 0 hata).
- Toplam 307 testle %100 doğrulandı.

### Faz 12: Ticari Monolitik ReRAM Çıkarım ASIC'i (TSMC 28nm/16nm Tapeout) 🔜
- 169 mm² monolitik zar, gömülü ReRAM hücreleri, PCIe Gen5 arayüzü, 0.28W sub-watt tüketim.
- HBM bağımsız, kurumsal ölçekte ekonomik ve ultra hızlı LLM çıkarım sunucuları.

---

## 📦 Sürüm Planı (Versioning Plan)

- **v0.1** — BF16 kanıtlandı, Gemma3 1B kayıpsız derlendi ✅
- **v0.2** — Amaranth 0.5 RTL donanım karoları, 0-DSP Yosys sentezi ve ECP5 P&R zamanlama analizi ✅
- **v0.3** — PPA ve Filo TCO modeli tamamlandı (Taalas vs RailNet ReRAM $33.9M avantajı) ✅
- **v0.4** — Çoklu model ailesi desteği (Llama-3/3.2, Qwen-2.5, Gemma-3), RoPE scaling, $O(1)$ KVCache (Adım 1) ✅
- **v0.5** — Hızlı C++/SIMD CPU çekirdeği ve çoklu iş parçacığı hızlandırması (Adım 2) ✅
- **v0.6** — INT8 ray paletleyicisi, karma hassasiyet (Mixed-Precision), C++ AVX2 INT8 çekirdeği ve Amaranth INT8 RTL karosu (%86.9 LUT tasarrufu) (Adım 3) ✅
- **v0.7** — Sanal donanım ve saykıl hassasiyetinde hızlandırıcı simülatörü ve telemetri motoru (Adım 4) ✅
- **v0.8** — Çoklu-Karo NoC, AXI4-Stream/Lite `RailNetTop`, 7,149 satır Verilog export ve Yosys sentezi (Faz 10) ✅
- **v0.9** — PCIe/FPGA Runtime Sürücüsü, SkyWater 130nm / Caravel ASIC Harness ve OpenLane akışı (Faz 11) ✅
- **v1.0** — Ticari üretime hazır ReRAM Weight-in-Silicon mimarisi ve Silikon Tapeout 🔜
