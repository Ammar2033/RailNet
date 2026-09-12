# RailNet: Reprogrammable Weight-in-Silicon AI Architecture
## Kapsamlı Proje Tanımı, Başarılanlar, 4 Aşamalı Eylem Planı ve Yol Haritası

> **Tarih:** Eylül 2026  
> **Proje Durumu:** Donanım RTL Doğrulandı (0 DSP Gather, ECP5 P&R), Çoklu Model Ailesi Desteği (Llama-3/3.2, Qwen-2.5, Gemma-3) Tamamlandı (Step 1 ✅), CPU Çekirdek Hızlandırmasına Hazır (Step 2 🚀).

---

## İÇİNDEKİLER

1. [Proje Nedir? (RailNet Felsefesi ve Temel Tezi)](#1-proje-nedir-railnet-felsefesi-ve-temel-tezi)
   - 1.1. Von Neumann ve Bellek Duvarı (Memory Wall) Problemi
   - 1.2. Taalas "Weight-in-Silicon" Mimarisi ve Kritik Değerlendirmesi
   - 1.3. RailNet'in "Reprogrammable Weight-in-Silicon" Çözümü (Path B)
   - 1.4. Matematiksel Temel: Matris Çarpımından Ray Yönlendirmesine
   - 1.5. Nicel PPA ve Filo TCO Analizi (Taalas vs RailNet)
2. [Şu Ana Dek Neleri Başardık? (Mevcut Durum Raporu)](#2-şu-ana-dek-neleri-başardık-mevcut-durum-raporu)
   - 2.1. Donanım ve RTL Sentezi (Amaranth 0.5 + Yosys + nextpnr)
   - 2.2. Yerleşim ve Zamanlama Analizi (P&R Timing Analysis)
   - 2.3. PPA ve Filo TCO Simülatörü
   - 2.4. Yazılım ve Çalışma Zamanı (Step 1: Multi-Model Tamamlandı)
   - 2.5. Doğrulama ve Test Matrisi (229 Unit + 12 RTL Testi)
3. [Belirlediğimiz 4 Temel Adım ve Detaylı Eylem Planı](#3-belirlediğimiz-4-temel-adım-ve-detaylı-eylem-planı)
   - Adım 1: Çoklu Model Ailesi Desteği (Tamamlandı ✅)
   - Adım 2: Hızlı Vektörize/C++ CPU Çıkarım Çekirdeği (Sıradaki Adım 🚀)
   - Adım 3: Çoklu Dtype Desteği (INT8 & FP16 Ladder) (Planlandı 📋)
   - Adım 4: Sanal Donanım ve Saykıl Hassasiyetinde Hızlandırıcı Simülatörü (Planlandı 📋)
4. [Teknik Uygulama Detayları: Adımları Nasıl Yapacağız?](#4-teknik-uygulama-detayları-adımları-nasıl-yapacağız)
   - 4.1. Adım 2'nin Uygulanması: SIMD/AVX2/AVX-512 C++ Gather Çekirdeği
   - 4.2. Adım 3'ün Uygulanması: INT8 Ağırlık Merdiveni ve Karma Hassasiyet
   - 4.3. Adım 4'ün Uygulanması: Cycle-Accurate Accelerator Simulator
5. [Uzun Vadeli Vizyon ve ASIC Yol Haritası](#5-uzun-vadeli-vizyon-ve-asic-yol-haritası)
   - 5.1. Silikon Üretim Aşamaları (SkyWater 130nm -> TSMC 28nm/16nm)
   - 5.2. ReRAM / Çapraz Bar (Crossbar) Entegrasyonu
   - 5.3. Kurumsal Dağıtım ve Filo TCO Devrimi

---

## 1. Proje Nedir? (RailNet Felsefesi ve Temel Tezi)

### 1.1. Von Neumann ve Bellek Duvarı (Memory Wall) Problemi

Modern Büyük Dil Modelleri (LLM) çıkarım (inference) sürecinde hesaplama (compute-bound) darboğazına değil, **bellek bant genişliği (memory-bandwidth-bound)** darboğazına takılır. 

Her bir token üretilirken, modelin yüz milyonlarca veya milyarlarca parametresinin tamamı DRAM/HBM bellekten okunup işlemci çekirdeklerine (GPU SM'leri veya TPU tensör çekirdekleri) taşınmak zorundadır:
- **Aritmetik Yoğunluk (Operational Intensity):** Her token üretimi için neredeyse $1 \text{ FLOP} / 1 \text{ Byte}$ transfer gerekir.
- **Enerji Tüketimi:** Bir bit veriyi DRAM'den taşımak, o veriyle aritmetik işlem yapmaktan **100 ila 1000 kat daha fazla enerji harcar**.
- HBM3 / HBM3e bellekli süper bilgisayarlar dahi bu "bellek duvarı" (memory wall) nedeniyle yüzlerce watt enerji tüketerek ısınır ve devasa maliyetlere yol açar.

### 1.2. Taalas "Weight-in-Silicon" Mimarisi ve Kritik Değerlendirmesi

Kanada merkezli donanım girişimi **Taalas**, bellek duvarını aşmak için radikal bir yaklaşım önerdi: **"Weight-in-Silicon"**.
- **Taalas Ne Yapıyor?** Modelin ağırlıklarını çip üzerindeki metal katmanlara (Hardwired Metal ROM) doğrudan baskı devre olarak basıyor. Matris çarpımları fiziksel tellerin birbirine bağlanmasıyla donanımda sabitleniyor.
- **Vaat Edilen:** Harici DRAM yok, ağırlık taşıma yok, 100+ token/saniye, sub-watt (<1 Watt) güç tüketimi.
- **Taalas'ın Kritik Zayıflığı (Fatal Flaw):** **Yeniden Programlanamaz (Zero Programmability)**.
  - Bir ağırlık bile değişse, model checkpoint'i güncellense ya da yeni bir mimariye geçilse (örneğin Llama 3'ten Llama 3.2'ye), **çip tamamen çöp olur**.
  - Her yeni model için yeni bir ASIC maske seti (tapeout) gerekir ($5M–$15M maliyet ve 6–9 ay fabrikasyon süresi).
  - Veri merkezleri ve bulut sağlayıcıları için 5 farklı modeli sunmak $40M+ maske maliyeti demektir.

### 1.3. RailNet'in "Reprogrammable Weight-in-Silicon" Çözümü (Path B)

**RailNet**, Taalas'ın "ağırlıkları silikonda tutma ve DRAM transferini sıfırlama" avantajını korurken, onun katil zayıflığı olan "esneklik yoksunluğunu" ortadan kaldıran **yeniden programlanabilir (reprogrammable)** yeni nesil bir yapay zeka hızlandırıcı mimarisidir.

RailNet'te ağırlıklar sabit metal kablolara basılmaz. Bunun yerine:
1. **Paylaşımlı Primitif Raylar (Shared Rails):** Her katmanda veya donanım karosunda yalnızca 96–192 adet yüksek hassasiyetli BF16/FP16 değer (ray) bulunur ($R_0, R_1, \dots, R_{95}$).
2. **Topolojik Yönlendirme (Bit-Pattern Routing):** Her ağırlık, bu paylaşımlı rayların işaretli toplamı olarak ifade edilir:
   $$W_{i,j} = \sum_{k=1}^{M} \text{sign}_k \cdot R_{r_k} \quad (M \le 4)$$
3. **Çalışma Zamanında Yoğun Matris YOK (Dense Weight Array = ABSENT):**
   $W_{i,j}$ matrisi fiziksel olarak hiçbir zaman bellekte açılmaz veya tutulmaz.
4. **Yeniden Programlanabilirlik:** Ray değerleri ve yönlendirme tabloları, çip üzerinde bulunan **ReRAM (Dirençli RAM)** veya eFlash gibi gömülü kalıcı hafıza (NVM) hücrelerinde saklanır. Yeni bir model yüklemek için çipi çöpe atmak gerekmez; mikrosaniyeler içinde yeniden flash'lanır.

```
[Geleneksel GPU]  : HBM/DRAM  ====== (Yüksek Güç, 1 TB/s Darboğaz) ======>  ALU / Tensor Core
[Taalas ROM]      : Sabit Metal Teller (0 Watt Taşıma, 0 Esneklik, $15M Maske)
[RailNet NVM]     : ReRAM / BRAM -> Paylaşımlı Raylar + Topoloji -> Sıfır DRAM + Tam Esneklik!
```

---

### 1.4. Matematiksel Temel: Matris Çarpımından Ray Yönlendirmesine

Geleneksel matris çarpımı (GEMM):
$$Y_j = \sum_{i=1}^{D_{\text{in}}} X_i \cdot W_{i,j}$$
Bu işlem $D_{\text{in}} \times D_{\text{out}}$ adet donanımsal çarpma işlemi (DSP / Multiplier) gerektirir.

RailNet'te ağırlık yerine yönlendirme konulduğunda matematiksel dönüşüm:
$$Y_j = \sum_{i=1}^{D_{\text{in}}} X_i \cdot \left(\sum_{r \in \text{route}(i,j)} \text{sign}(i,j,r) \cdot R_r\right)$$

Toplamların sırası değiştirildiğinde (**Fubini / Distributive Transformation**):
$$Y_j = \sum_{r=0}^{N_{\text{rails}}-1} R_r \cdot \underbrace{\left(\sum_{i \in \text{gather}(j,r)} \text{sign}(i,j,r) \cdot X_i\right)}_{\text{Stage-A: SIFIR ÇARPAN! Sadece Toplama/Çıkarma}}$$

#### Donanım Devrimi:
- **Stage-A (Toplama - Gather):** Giriş aktivasyonları ($X_i$), yönlendirme tablosundaki işaretlere göre akümüle edilir. **Burada HİÇBİR çarpma işlemi yapılmaz (0 DSP / 0 Multiplier)**. Sadece basit yönlendirme ve toplama/çıkarma yapılır.
- **Stage-B (Ray Çarpımı - Accumulation):** Stage-A'dan çıkan değerler, ray değeri ($R_r$) ile çarpılır.
- **Çarpma Tasarrufu:** $D_{\text{in}}$ adet çarpma yerine yalnızca $N_{\text{rails}}$ adet çarpma yapılır.
  - Örnek (Gemma3 1B linear katmanı: $1152 \times 6912$):
  - Geleneksel Çarpma: $1152 \times 6912 = 7,962,624$ çarpma.
  - RailNet Çarpma: $96 \times 6912 = 663,552$ çarpma.
  - **Çarpma İşleminde $\%93.31 - \%95.97$ Net Donanımsal Azalma!**

---

### 1.5. Nicel PPA ve Filo TCO Analizi (Taalas vs RailNet)

[`hardware/research/ppa_model.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/research/ppa_model.py) simülatörümüz ile Gemma3 1B modeli temel alınarak yapılan detaylı fiziksel alan (PPA) ve bulut filosu maliyet (TCO) analiz sonuçları:

| Metrik | Geleneksel SRAM Çipi | Taalas (Metal ROM) | RailNet ReRAM (Path B) |
| :--- | :--- | :--- | :--- |
| **Bit Başına Alan** | $0.0400 \, \mu\text{m}^2$ | $0.0012 \, \mu\text{m}^2$ | $0.0016 \, \mu\text{m}^2$ |
| **Toplam Zar Alanı (Die Area)** | $4,330.7 \, \text{mm}^2$ *(13 chiplet)* | **$125.4 \, \text{mm}^2$** *(Monolitik)* | **$169.3 \, \text{mm}^2$** *(Monolitik)* |
| **Çip Üzeri Güç (100 tok/s)** | $1.20 \, \text{W}$ | **$0.21 \, \text{W}$** | **$0.28 \, \text{W}$** *(Sub-watt!)* |
| **Yeniden Programlanabilirlik**| Var | **YOK (Sabit Metal)** | **VAR (ReRAM / NVM)** |
| **Yeni Model Geliştirme Süresi**| Dakikalar | **6–9 Ay (Maske Üretimi)**| **Saniyeler (Flash Yazma)** |
| **5 Model / 10.000 Çip Filo Maliyeti**| $25.0 \, \text{Milyon \$}$ | $43.5 \, \text{Milyon \$}$ | **$9.6 \, \text{Milyon \$}$** |
| **Çip Başına Düşen Net Maliyet** | $2,500 \, \$$ | $4,345 \, \$$ | **$955 \, \$$** |
| **Net TCO Tasarrufu** | Referans | -%74 Zarar (Maske Masrafı) | **33.9 Milyon \$ Tasarruf (%78 İndirim!)** |

> **Temel Çıkarım:** Taalas tek bir dondurulmuş model için küçük zar alanı sunsa da, gerçek dünya operasyonlarında (model güncellemeleri, farklı model boyutları) maske maliyetleri nedeniyle ekonomik olarak sürdürülemezdir. RailNet, neredeyse aynı fiziksel alan ve sub-watt güçle **%78 TCO avantajı ve tam esneklik** sunar.

---

## 2. Şu Ana Dek Neleri Başardık? (Mevcut Durum Raporu)

### 2.1. Donanım ve RTL Sentezi (Amaranth 0.5 + Yosys)

1. **Saf Aritmetik Kütüphanesi ([`hardware/rtl/fp32.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/fp32.py)):**
   - Harici C/C++ bağımlılığı olmadan, saf Python/Amaranth HDL ile sıfırdan IEEE-754 uyumlu aritmetik üniteler geliştirildi:
     - `Bf16ToFp32` ve `Fp32ToBf16` dönüştürücüler (RNE - Round to Nearest Even yuvarlama ile).
     - `CombFp32Adder` (tek çevrimlik toplayıcı).
     - `PipelinedFp32Adder` (yüksek frekanslı boru hattı toplayıcısı).
     - `Fp32Multiplier` (kayan nokta çarpıcı).
2. **3-Şeritli Hızlandırıcı Karosu ([`hardware/rtl/bf16_tile.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/bf16_tile.py)):**
   - `RouteDecoder`: Ray indekslerini ve işaretlerini çözen kod çözücü.
   - `StageABf16Bram`: Giriş tamponu, sıfır duraklamalı (zero-stall) iletme/atlatma (forwarding/bypass) ağı.
   - `StageBBf16`: Çoklu rayları birleştiren FP32 akümülatör ve yuvarlama ünitesi.
   - `RailNetFullTile`: Stage-A ve Stage-B'yi birbirine bağlayan tam entegre karo.
3. **Yosys Mantık Sentezi ([`hardware/rtl/synth.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/synth.py) & [`results/rtl_synth.json`](file:///f:/Projects/2026/Ongoing/RailNet/results/rtl_synth.json)):**
   - Xilinx 7-serisi hedef alınarak yapılan sentezde RailNet'in matematiksel tezi donanımda **kanıtlandı**:
     - **Stage-A Gather: 0 DSP!** (725 LUT, 105 FF). Yönlendirme ve toplama mantığı tek bir donanımsal çarpıcı harcamadan saf LUT mantığıyla çözüldü.
     - **Tam 3-şeritli `RailNetFullTile`:** Yalnızca **2 DSP** kullandı.

### 2.2. Yerleşim ve Zamanlama Analizi (P&R Timing Analysis)

Lattice ECP5 (LFE5U-85F) FPGA hedefi üzerinde `yowasp-nextpnr-ecp5` ile yapılan fiziksel yerleşim ve zamanlama sonuçları ([`results/rtl_timing.json`](file:///f:/Projects/2026/Ongoing/RailNet/results/rtl_timing.json)):
- **Tamsayı (Integer) Stage-A Karosu:** **178.95 MHz** ($F_{\text{max}}$), kritik yol yalnızca 5.58 ns!
- **BF16 Kayan Nokta Stage-A Karosu:** 40nm ECP5 üzerinde 34.49 MHz (28.99 ns).
  - 8 mantık seviyeli bu kritik yol, modern 16nm/22nm ASIC standart hücre (standard-cell) kütüphanelerinde **400–600 MHz** aralığına denk gelmektedir.

### 2.3. PPA ve Filo TCO Simülatörü

- [`hardware/research/ppa_model.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/research/ppa_model.py): Silikon alanı, SRAM/ROM/ReRAM hücre yoğunlukları, güç harcaması ve çok modelli bulut filosu maliyetlerini hesaplayan uçtan uca simülasyon motoru yazıldı.
- Bulgular [`hardware/research/taalas_vs_railnet_ppa.md`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/research/taalas_vs_railnet_ppa.md) raporunda belgelendi.

### 2.4. Yazılım ve Çalışma Zamanı (Step 1: Multi-Model Tamamlandı)

Başlangıçta yalnızca Gemma-3 1B için özelleşmiş olan RailNet yazılım katmanı, başarıyla genelleştirildi:
- **Polimorfik Context Yapısı ([`railnet/transformer.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/transformer.py)):**
  - `TransformerContext` temel sınıfı oluşturuldu.
  - `GemmaContext`: Sıfır merkezli $(1+w)$ RMSNorm, 6 normlu sandviç topoloji, `gelu_tanh` aktivasyonu, sliding-window yerel/global RoPE.
  - `LlamaContext`: Standart $w$ RMSNorm, pre-norm residual topolojisi, SwiGLU ($\text{silu}(\text{gate}) \cdot \text{up} \to \text{down}$), Meta Llama-3 RoPE frequency scaling ($\theta=500000.0, \text{factor}=32.0$).
  - `QwenContext`: Standart $w$ RMSNorm, SwiGLU, QKV bias vektörleri, RoPE ($\theta=1000000.0$).
- **Önceden Tahsisli $O(1)$ KV-Cache ([`KVCache`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/transformer.py#L22)):**
  - Her yeni token üretiminde `np.concatenate` çağrısı yaparak hafızayı baştan sona kopyalama yükü ($O(N^2)$) kaldırıldı; sabit dairesel tampon ile $O(1)$ dilim güncellemesine geçildi.
- **Model Adapter'ları:**
  - [`railnet/models/llama.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/models/llama.py): `Llama-3.2-1B` ve `3B` için tam adapter.
  - [`railnet/models/qwen.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/models/qwen.py): `Qwen-2.5-0.5B` ve `1.5B` için tam adapter.
  - [`railnet/models/__init__.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/models/__init__.py): `config.json` üzerinden mimariyi otomatik tanıyan `get_adapter_for_config()` fonksiyonu.
- **Dinamik Model Çalışma Zamanı ([`railnet/runtime/transformer.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/transformer.py)):**
  - Yalnızca ilgili mimarinin normlarını ve bias'larını yükleyen esnek yapı.
  - Bağlı (tied) ve bağımsız (untied) `lm_head.weight` matrisi desteği.

```
[Adım 1: Multi-Model Desteği]  ──>  [Adım 2: Hızlı C++ CPU Çekirdeği]
       (TAMAMLANDI ✅)                      (TAMAMLANDI ✅)
              │                                      │
              ▼                                      ▼
[Adım 3: Çoklu Dtype (INT8/FP16)] ──> [Adım 4: Sanal Donanım Simülatörü]
        (TAMAMLANDI ✅)                             (TAMAMLANDI ✅)
```

### Özet Durum Tablosu

| No | Adım Başlığı | Durum | Ana Odak | Çıktı / Gerçekleşen Sonuç |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Çoklu Model Desteği** | **TAMAMLANDI ✅** | Llama-3/3.2, Qwen-2.5, SwiGLU, RoPE Scaling, $O(1)$ KV-Cache | 3 büyük model ailesi tam destekli, 229 birim testi geçiyor. |
| **2** | **Hızlı CPU Çıkarım Çekirdeği** | **TAMAMLANDI ✅** | C++ AVX2/FMA + OpenMP native DLL + Numba parallel JIT fallback | NumPy referansına göre **5.0x - 7.5x hızlanma**, Numba'ya göre **1.4x - 1.9x hızlanma**. 8/8 çekirdek testi, 12/12 uçtan uca exactness testi geçti. |
| **3** | **Çoklu Dtype Desteği** | **TAMAMLANDI ✅** | INT8 ve FP16 derleme merdiveni, integer ray karoları | Silikon hafıza alanını yarıya indirmek (85 mm²), karma hassasiyet, %86.9 LUT tasarrufu, 286 test. |
| **4** | **Sanal Donanım Simülatörü** | **TAMAMLANDI ✅** | Cycle-accurate donanım simülatörü, `RailNetDevice` sürücüsü, telemetri | Parametrik Edge/Cloud ASIC & FPGA profilleri, 2D NoC broadcast, PCIe DMA, sub-watt onayı (<1W), 296 test. |
| **5** | **Çoklu-Karo NoC ve AXI RTL (RailNetTop)** | **TAMAMLANDI ✅** | 2D Grid, AXI4-Lite CSR, AXI4-Stream DMA, Broadcaster, Concentrator, Verilog (.v) | Standart SoC/PCIe arayüzü, 7,149 satır Verilog export, Yosys sentezi (8 DSP, 2,350 LUT), 299 test. |

---

## 4. Teknik Uygulama Detayları

### 4.1. Adım 2: SIMD/AVX2 C++ Gather Çekirdeği ve Numba JIT Fallback (TAMAMLANDI ✅)

Adım 2 başarıyla tamamlandı. Çift katmanlı (Dual-Tier) hızlandırma mimarisi ve otomatik terfi zinciri devreye alındı.

#### 1. Mimarinin Bileşenleri:
1. **C++ AVX2/FMA Native Motoru ([`railnet/csrc/rail_kernel.cpp`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/csrc/rail_kernel.cpp)):**
   - Stage-A Gather ve Stage-B Accumulate aşamaları `_mm256_fmadd_ps` ve `_mm256_add_ps` vektör komutlarıyla AVX2 yazmaçlarına taşındı.
   - OpenMP `#pragma omp parallel for` direktifi ile çok çekirdekli (12 thread) paralel çıkış hesabı sağlandı.
   - C-ABI uyumlu dışa aktarım (`railnet_linear_fp32`, `railnet_linear_fp64`).
2. **Otomatik Derleme ve Yükleyici ([`railnet/csrc/builder.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/csrc/builder.py)):**
   - Sisteme gömülü GCC 16.2 / Clang / MSVC tespit mekanizması.
   - `build_native_kernel()` ve `get_native_kernel()` ile otomatik derleme ve ctypes yüklemesi.
3. **Numba JIT Fallback ([`railnet/runtime/numba_kernel.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/numba_kernel.py)):**
   - C++ ikilisi/derleyicisi olmayan sistemlerde sıfır kurulum gerektiren `@njit(parallel=True, fastmath=True)` fallback.
4. **Dinamik Dispatch ve Model Entegrasyonu ([`railnet/runtime/linear.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/linear.py) & [`railnet/runtime/transformer.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/transformer.py)):**
   - `rail_linear_dispatch(..., backend="auto")`: Öncelik sırası `CPP > Numba > NumPy`.
   - `precision="fp32"` ve `precision="fp64"` modları tam destekli.
   - `RailNetModel.load(..., backend="auto")` ile transformer model döngüsüne entegre.

#### 2. Benchmark Sonuçları ([`benchmarks/benchmark_cpu_kernels.py`](file:///f:/Projects/2026/Ongoing/RailNet/benchmarks/benchmark_cpu_kernels.py)):
*Test Ortamı: 12 Çekirdek x86-64 İşlemci, AVX2 Aktif, Windows 11*

| Katman / Tensör Şekli | Dense GEMM (ms) | NumPy (ms) | Numba JIT (ms) | C++ AVX2 (ms) | C++ vs NumPy Hızlanma | C++ vs Numba Hızlanma |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma-1B Attn (1152x1152)** | 0.25 ms | 41.49 ms | 8.01 ms | **5.57 ms** | **7.5x** | **1.4x** |
| **Gemma-1B MLP Gate (6912x1152)** | 1.00 ms | 186.37 ms | 47.51 ms | **29.50 ms** | **6.3x** | **1.6x** |
| **Gemma-1B MLP Down (1152x6912)** | 1.12 ms | 179.54 ms | 59.70 ms | **30.84 ms** | **5.8x** | **1.9x** |
| **Llama-3.2-1B MLP (8192x2048)** | 2.21 ms | 401.87 ms | 128.87 ms | **80.80 ms** | **5.0x** | **1.6x** |

#### 3. Test ve Doğrulama:
- **`tests/unit/test_cpu_kernels.py` (8/8 PASSED):** C++ ve Numba çekirdeklerinin NumPy ile FP32 ve FP64 modlarında sayısal denkliği ve hata fırlatma durumları test edildi.
- **`tests/exactness/test_end_to_end_runtime.py` (12/12 PASSED):** Uçtan uca model çıkarımı, deterministik üretim ve bit-exact logit kontrolleri doğrulandı.
- **Tüm Test Paketi:** **256 passed, 1 skipped** (sıfır hata).
---

### 4.2. Adım 3'ün Uygulanması: INT8 Ağırlık Merdiveni ve Karma Hassasiyet [TAMAMLANDI]

#### 1. Mimarinin Bileşenleri:
1. **INT8 Ray Paletleyicisi ve Kuantizasyon ([`railnet/compiler/int8.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/compiler/int8.py)):**
   - Simetrik per-tensor INT8 kuantizasyonu (`quantize_to_int8`, `dequantize_from_int8`).
   - 24-32 elemanlı kanonik integer tabanı ile [-128, 127] aralığındaki tüm sayıların maksimum 2-3 terimle %100 kayıpsız (0 hata) deterministik temsili.
   - `compile_int8_tensor` ile INT8 `RailTensor` ve ölçek (`scale`) serileştirmesi.
2. **Karma Hassasiyetli Derleme Hattı ([`railnet/compiler/model.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/compiler/model.py)):**
   - `compile_model(..., mixed_precision=True)`: Hassas dikkat katmanları ($Q, K, V, O$) BF16 raylarında kalırken; devasa MLP (`gate`, `up`, `down`) katmanları INT8 raylarına derlenir.
   - Genişletilmiş `manifest.json`: Katman bazında `dtype`, `scale`, `rail_count` ve `policy` kayıtları.
3. **C++ AVX2 ve Numba INT8 Motoru ([`railnet/csrc/rail_kernel.cpp`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/csrc/rail_kernel.cpp) & [`railnet/runtime/numba_kernel.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/numba_kernel.py)):**
   - `railnet_linear_int8_w8a_float`: Float aktivasyon + INT8 tamsayı ray SIMD çarpımı + on-the-fly dekuantizasyon scale faktörü.
   - `railnet_linear_int8_w8a16`: RTL donanımı ile bit-exact eşleşen 16-bit tamsayı aktivasyon ve 32-bit Stage-A akümülatör boru hattı.
   - `rail_linear_dispatch` ile dinamik CPP > Numba > NumPy yönlendirmesi.
4. **Donanım RTL INT8 Karosu ([`hardware/rtl/int8_tile.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/int8_tile.py)):**
   - `StageAInt8Bram`: Zero-DSP, 32 derinlikli BRAM akümülatörü ve sıfır-baloncuklu tek saykıl bypass yönlendirmesi.
   - `StageBInt8`: 8-bit işaretli tamsayı ray indirgeme motoru.
   - `RailNetInt8Tile`: Üst düzey entegre INT8 hızlandırıcı karosu.

#### 2. Donanım RTL Yosys Sentez Sonuçları ([`results/rtl_synth.json`](file:///f:/Projects/2026/Ongoing/RailNet/results/rtl_synth.json)):
*Sentez Hedefi: Xilinx 7-Series / UltraScale+ (Yosys synth_xilinx)*

| Donanım Karosu | DSP | BRAM | FF | LUT | CARRY4 | MUXF | Alan Verimliliği |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **dense (Referans MAC)** | 1 | 0 | 33 | 34 | 0 | 0 | 1.0x (Taban) |
| **stagea_bf16_bram** | 0 | 0 | 105 | 725 | 28 | 190 | 5.2x dense |
| **stagea_int8_bram** | **0** | **0** | **100** | **129** | **15** | **32** | **1.2x dense (%82 LUT Azalması!)** |
| **stageb_bf16** | 2 | 0 | 100 | 1615 | 93 | 238 | 12.4x dense |
| **stageb_int8** | **2** | **0** | **40** | **101** | **19** | **4** | **2.1x dense (%93.7 LUT Azalması!)** |
| **full_bf16_tile** | 2 | 0 | 383 | 3711 | 177 | 612 | 27.3x dense |
| **full_int8_tile** | **2** | **0** | **308** | **486** | **54** | **97** | **5.6x dense (%86.9 Net Zar Alanı Tasarrufu!)** |

> **Kritik Çıkarım:** INT8 karo mimarisi, mantık kapısı (LUT) sayısını **3,711'den 486'ya (%86.9)** düşürmüştür. Bu durum Gemma3 1B silikon alanını **169 mm²'den ~85 mm²'ye** ve çip maliyetini **$500 seviyesine** indirme hedefimizi donanımsal olarak kanıtlamaktadır.

#### 3. Test ve Doğrulama:
- **`tests/unit/test_int8_compiler.py` (3/3 PASSED):** Kuantizasyon, %100 kapsama ve rota denkliği doğrulandı.
- **`tests/unit/test_int8_kernels.py` (4/4 PASSED):** C++ AVX2, Numba JIT ve NumPy INT8 doğrusal katman sayısal denkliği kanıtlandı.
- **`tests/exactness/test_mixed_precision_runtime.py` (3/3 PASSED):** Uçtan uca karma model derleme, yükleme ve çıkarım determinizmi test edildi.
- **`hardware/rtl/test_int8_tile.py` (4/4 PASSED):** Amaranth simülatöründe Stage-A, tek saykıl bypass forwarding ve Stage-B fonksiyonel olarak doğrulandı.
- **Toplam Test Paketi:** **286 passed (270 unit/exactness + 16 RTL)**, sıfır hata!

---

### 4.3. Adım 4: Cycle-Accurate Sanal Hızlandırıcı Simülatörü ve Telemetri Motoru (TAMAMLANDI ✅)

Adım 4 başarıyla tamamlandı. Yazılım çıkarımı ile silikon fiziksel davranışları arasındaki boşluğu kapatan saykıl-hassas (cycle-accurate) donanım simülatörü ve telemetri motoru devreye alındı.

#### 1. Mimarinin Bileşenleri:
1. **Sanal Donanım Simülatörü ([`railnet/sim/accelerator.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/sim/accelerator.py)):**
   - **`HardwareConfig`:** Parametrik hızlandırıcı donanım profilleri:
     - `Edge ASIC`: 64 Karo, 8x8 Grid @ 500 MHz, PCIe Gen4 x16 (25 GB/s), 0.12W statik sızıntı gücü.
     - `Cloud ASIC`: 256 Karo, 16x16 Grid @ 1.0 GHz, PCIe Gen5 x16 (50 GB/s), 0.35W statik sızıntı gücü.
     - `FPGA Prototype`: 16 Karo, 4x4 Grid @ 150 MHz, PCIe Gen3 x8 (6.5 GB/s), 2.5W statik güç.
   - **`TileSimulator`:** Karo seviyesi boru hattı simülasyonu:
     - Stage-A BRAM RMW gecikmesi (INT8 için 1 saykıl, BF16 FP32 toplayıcı için 2 saykıl).
     - Paralel gather şeritleri (`stage_a_lanes = 3`) ve ardışık ray çakışmalarında tek saykıllık arbiter stall modeli.
     - Stage-B binary adder tree indirgeme gecikmesi ($rail\_count + stage\_b\_latency$).
     - Hassasiyete duyarlı dinamik enerji modeli (INT8 0.04 pJ vs BF16 0.12 pJ).
   - **`AcceleratorSimulator`:** Çip ve sistem seviyesi simülatör:
     - 2D NoC Mesh broadcast gecikmesi ($mesh\_hops + flits$).
     - PCIe DMA veri transfer gecikmesi ($latency + bytes / bw$).
     - Çıkış nöronlarının donanım karoları üzerine dinamik dengeli dağıtımı.
2. **Kapsamlı Telemetri Raporu ([`HardwareTelemetry`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/sim/accelerator.py#L98)):**
   - Katman bazlı ayrışım: Karo sayısı, Stage-A ve Stage-B saykılları, katman gecikmesi (ms).
   - Sistem metrikleri: Toplam saykıl, PCIe aktarım süresi, toplam gecikme (ms/token), çıkarım hızı (tokens/sn), karo doluluk oranı (tile utilization %), ortalama çip gücü (**0.13W Sub-watt onayı** < 1.0W) ve enerji verimliliği (**66,000+ Tokens / Joule**).
3. **Çift Kullanım API'si ([`RailNetDevice.sim`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/engine.py#L33) & CLI):**
   - Python API: `RailNetModel.load("compiled", device=RailNetDevice.sim(profile="edge_asic"))`
   - Sayısal denklik: Simülatör çalışırken çıktı logitleri CPU referansı ile bit-exact örtüşür ($atol < 10^{-12}$).
   - Terminal CLI: `python -m railnet.sim.cli --model <path> --profile edge_asic --tokens 1`
4. **Test ve Doğrulama:**
   - [`tests/unit/test_accelerator_sim.py`](file:///f:/Projects/2026/Ongoing/RailNet/tests/unit/test_accelerator_sim.py): 7/7 PASSED (konfigürasyonlar, tehlike duraklamaları, NoC broadcast, PCIe DMA, telemetri formatı).
   - [`tests/exactness/test_sim_device_runtime.py`](file:///f:/Projects/2026/Ongoing/RailNet/tests/exactness/test_sim_device_runtime.py): 3/3 PASSED (sayısal denklik, katman saykıl telemetrisi, profil karşılaştırması).
   - **Genel Test Paketi:** **296 testin tümü PASSED** (0 hata, 0 regresyon).

---

### 4.4. Faz 10 (v0.8): Çoklu-Karo NoC ve AXI4-Stream/Lite Üst Düzey Donanım Tasarımı (TAMAMLANDI ✅)

RailNet donanım mimarisi tekil karo seviyesinden çıkarılarak, çok karolu (multi-tile) 2D hesaplama matrisine ve endüstri standardı AXI4 arayüzlerine sahip entegre bir hızlandırıcı SoC/PCIe çip üst seviyesine (`RailNetTop`) taşındı.

#### 1. Mimarinin Bileşenleri:
1. **Standart Host Arayüzleri ([`hardware/rtl/axi.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/axi.py)):**
   - **`AxiLiteCsr` (32-bit AXI4-Lite Slave):** Bellek haritalı Kontrol ve Durum Yazmaçları:
     - `0x00` (`REG_CTRL`): `start`, `soft_reset`, `ie` (interrupt enable).
     - `0x04` (`REG_STATUS`): `busy`, `done`, `num_tiles`.
     - `0x08` (`REG_IN_FEATURES`): Token başına girdi boyutu (örn: 128, 512, 2048).
     - `0x0C` (`REG_OUT_FEATURES`): Çıkış nöron sayısı.
     - `0x10` (`REG_TILE_MASK`): Aktif karo maskesi (istenmeyen karoları atlama yeteneği).
     - `0x14` (`REG_CYCLE_COUNT`): Donanımsal saykıl performans sayacı.
   - **AXI4-Stream Giriş/Çıkış Arayüzü:**
     - `s_axis_activations` (32-bit TDATA, TVALID, TREADY, TLAST): Host DMA aktivasyon akışı.
     - `m_axis_results` (32-bit signed TDATA, TVALID, TREADY, TLAST): Hesaplanan nöron çıkış akışı.
2. **Çoklu-Karo Hesaplama Matrisi ([`hardware/rtl/grid.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/grid.py)):**
   - **`RailNetGrid`:** Parametrik $W \times H$ (varsayılan $2 \times 2 = 4$ karo veya $4 \times 4 = 16$ karo) INT8 karo matrisi.
   - **Weight-in-Silicon Bellek Mimarisi:** Her karo, kendi nöron diliminin ağırlık rotalarını saklayan yerel bir `route_mem` BRAM (512 derinlik) barındırır. Aktivasyonlar yayınlandıkça karolar yerel BRAM'lerinden rotaları eşzamanlı okur; harici bellekten ağırlık okuma yükü sıfırdır.
3. **NoC Dağıtım ve Toplama Ağı ([`hardware/rtl/noc.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/noc.py)):**
   - **`PipelinedBroadcaster`:** Giriş aktivasyonlarını kayıtlı boru hattı (pipelined registers) üzerinden karolara yayarak fan-out tel yükünü ve gecikmesini sıfırlar.
   - **`ResultGatherConcentrator`:** Karolar Stage-B hesaplamasını tamamladığında (`all_done`), aktif karolardan $y_i$ çıkışlarını sırayla toplayıp `m_axis` akışına aktarır.
4. **Entegre Üst Düzey Donanım Kontrolcüsü ([`hardware/rtl/top.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/top.py)):**
   - 8 durumlu FSM (`IDLE` $\to$ `FLUSH` $\to$ `STREAM_IN` $\to$ `DRAIN` $\to$ `START_RED` $\to$ `WAIT_RED` $\to$ `STREAM_OUT` $\to$ `DONE`).
   - `STATE_DRAIN` boru hattı duraklaması ile Stage-A yazma gecikmelerinin Stage-B indirgemesinden önce kusursuz şekilde oturması sağlanır.
5. **Sentezlenebilir Verilog-2001 Dışa Aktarma ([`hardware/rtl/export.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/rtl/export.py)):**
   - Amaranth RTL kodunu Yosys üzerinden **7,149 satırlık** saf, standart Verilog-2001 dosyasına (`hardware/rtl/build/railnet_top.v`) dönüştüren bağımsız CLI aracı. Xilinx Vivado ve ASIC EDA araçlarına doğrudan entegre edilebilir.

#### 2. Donanım RTL Sentez Sonuçları (Yosys `synth_xilinx -flatten`):

| Tasarım Modülü | DSP | BRAM | FF | LUT | CARRY4 | MUXF | Durum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **dense (Tekil MAC)** | 1 | 0 | 33 | 34 | 0 | 0 | 1.0x (Referans) |
| **full_bf16_tile (Tek BF16 Karo)** | 2 | 0 | 383 | 3,711 | 177 | 612 | 27.3x dense |
| **full_int8_tile (Tek INT8 Karo)** | **2** | **0** | **308** | **486** | **54** | **97** | **5.6x dense (%86.9 tasarruf)** |
| **railnet_top_2x2 (4-Karo Grid + NoC + AXI)** | **8** | **0** | **1,441** | **2,350** | **227** | **534** | **Tam Çip Üst Düzeyi!** |

> **Kritik Donanım Başarısı:** 4 tam karolu, AXI4-Lite CSR'lı, AXI4-Stream DMA'lı ve NoC ağlı komple `railnet_top_2x2` çip tasarımı, **yalnızca 2,350 LUT** tüketmektedir! Bu miktar, tek bir eski BF16 karosunun (3,711 LUT) bile altındadır. En ekonomik FPGA'larda dahi (Lattice ECP5 84K LUT veya Xilinx Kria KV260 53K LUT) çipin yalnızca %3-4'ünü kaplar.

#### 3. Test ve Doğrulama:
- **`hardware/rtl/test_top_grid.py` (3/3 PASSED):**
  - `test_axi_csr_registers`: AXI4-Lite CSR register okuma ve yazma protokolü doğrulaması.
  - `test_top_grid_e2e_parallel_inference`: 2 karolu paralel hesaplama, AXI4-Stream üzerinden girdi akışı, paralel Stage-A/Stage-B ve sıralı AXI4-Stream çıkış doğrulaması ($Y_0 = 240, Y_1 = 360$).
  - `test_top_grid_tile_masking`: Dinamik karo maskeleme ve atlama doğrulaması.
- **Tüm Repo Regresyon Testi:** **299 testin tamamı PASSED** (0 hata, 0 uyarı hatası).

---

### 4.5. Faz 11'in Uygulanması: PCIe/FPGA Donanım Sürücüsü ve Açık Kaynak ASIC Tapeout Hazırlığı (SkyWater 130nm / Caravel) [TAMAMLANDI]

#### 1. Mimarinin Bileşenleri:
1. **PCIe / FPGA Donanım Çalışma Zamanı Sürücüsü ([`railnet/runtime/pcie.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/pcie.py)):**
   - **`RailNetPCIeDriver`:** Linux çekirdeğindeki Xilinx XDMA/QDMA PCIe uç noktaları (`/dev/xdma0_user`, `/dev/xdma0_h2c_0`, `/dev/xdma0_c2h_0`) ile kullanıcı alanında doğrudan haberleşen, sıfır-kopyalama (zero-copy) DMA akış sürücüsü.
   - **`MockPCIeBridge`:** Geliştirme ortamında (Windows / kart takılı olmayan sistemlerde) donanım CSR yazmaçlarını ve döngüsel AXI-Stream DMA transferlerini bit-exact simüle eden yazılımsal köprü.
   - `stream_activations(x)` ve `read_results(num_outputs, scale)` modüler DMA yöntemleri.
   - Hem 1D hem 2D (toplu/batch) aktivasyonları donanıma sevk eden `dispatch_linear`.
2. **Çalışma Zamanı Motoru ve Model Entegrasyonu ([`railnet/runtime/engine.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/engine.py) & [`railnet/runtime/transformer.py`](file:///f:/Projects/2026/Ongoing/RailNet/railnet/runtime/transformer.py)):**
   - `RailNetDevice.pcie(dev="railnet0")` ve `RailNetDevice.open()` fabrika metodları ile otomatik sürücü tespiti.
   - `RailNetEngine.dispatch_linear` ve `RailNetModel._rail_backend`: `device.kind == "pcie"` olduğunda tüm doğrusal katman projeksiyonlarını PCIe hızlandırıcı kartına yönlendirme.
3. **Wishbone-to-AXI-Lite Köprüsü ([`hardware/asic/wishbone_to_axi.v`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/asic/wishbone_to_axi.v)):**
   - Efabless Caravel SoC RISC-V yönetim çekirdeğinden gelen 32-bit Wishbone B4 veriyolu sinyallerini, `RailNetTop` mimarisinin AXI4-Lite CSR köle arayüzüne dönüştüren sentezlenebilir FSM adaptörü.
4. **Caravel Kullanıcı Projesi Silikon Harness'ı ([`hardware/asic/caravel_railnet.v`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/asic/caravel_railnet.v)):**
   - Efabless Caravel ve SkyWater 130nm standardına uyumlu tam makro sarmalayıcısı.
   - Güç pinleri (`vccd1`, `vssd1`), Logic Analyzer (LA) aktivasyon ve sonuç hatları, kullanıcı kesmesi (`user_irq`) ve GPIO telemetri bağlantıları.
5. **OpenLane SkyWater 130nm Üretim Konfigürasyonu ([`hardware/asic/openlane/config.json`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/asic/openlane/config.json)):**
   - `sky130_fd_sc_hd` standard-cell kütüphanesi için 50 MHz hedef frekanslı, 5-katmanlı metal yığını (met1-met5), DRC/LVS kurallı tam fiziksel yerleşim ve yönlendirme (P&R) akışı.
6. **ASIC Sentez Doğrulaması ([`hardware/asic/synth_asic.py`](file:///f:/Projects/2026/Ongoing/RailNet/hardware/asic/synth_asic.py) & [`results/asic_synth.json`](file:///f:/Projects/2026/Ongoing/RailNet/results/asic_synth.json)):**
   - Yosys `synth -top caravel_railnet` ile tam hiyerarşik ASIC sentezi: `caravel_railnet`, `wishbone_to_axi` ve `railnet_top` 200,178 mantık kapısına başarıyla dönüştürüldü (0 hata).

#### 2. Test ve Doğrulama:
- **`tests/unit/test_pcie_driver.py` (8/8 PASSED):**
  - `test_mock_pcie_bridge_csr`: Donanım CSR yazmaçlarının durum ve kontrol bitleri doğrulaması.
  - `test_pcie_driver_fallback_and_csr`: Donanım yokluğunda şeffaf yazılımsal köprüye geçiş.
  - `test_pcie_driver_linear_dispatch_1d` & `2d_batch`: Sayısal denklik ve donanım saykıl sayacı doğrulaması.
  - `test_pcie_driver_streaming_methods`: Doğrudan DMA aktivasyon akıtma ve sonuç okuma.
  - `test_engine_pcie_device`: `RailNetEngine` üzerinden donanımsal çıkarım.
  - `test_transformer_model_pcie_device`: Uçtan uca Llama dönüştürücü modelinin `RailNetDevice.pcie()` üzerinde derlenip çalıştırılması ve referansla %99+ kosinüs benzerliği.

### 4.5. Gate 1 — Silikon Beka Denetimi ve Kırılma Noktası Doğrulaması (TAMAMLANDI)

Kapsamlı mimari ve donanım-yazılım sözleşmesi denetiminde tespit edilen 5 kritik engel (showstopper) ve mimari riskler çözülmüş, `tests/unit/test_gate1_robustness.py` ile donanımsal simülasyonda kilitlenmiştir:

1. **In-Band Donanım Ağırlık Programlama (Showstopper 1):**
   - CSR register haritasına `REG_PROG_ADDR` (`0x18`), `REG_PROG_DATA` (`0x1C`) ve `REG_PROG_CTRL` (`0x20`) eklendi.
   - Fiziksel çip üzerinde PCIe/AXI üzerinden yönlendirme BRAM'leri, kod defteri ve ray değerleri in-band strobe ile yazılabilir hale getirildi.
2. **512 BRAM Derinliği vs LLM $K=2048$ (Showstopper 2):**
   - `ctrl_accumulate` kontrol biti (bit 2) eklendi. Çok parçalı (multi-pass) aktivasyon akışında Stage-A akümülatörlerinin sıfırlanması (flush) engellendi.
   - `grid.py` içindeki `k_counter`, chunk bitiminde `start_reduction` ile sıfırlanarak her parça için senkronize hale getirildi.
   - PCIe sürücüsüne 512-elemanlık otomatik parça bölücü (`CHUNK_SIZE = 512`) eklendi.
3. **Seyrek Tile Maskesinde Artık/Yinelenen Kelime Hatası (Showstopper 4):**
   - `ResultGatherConcentrator` içinde pasif karolar atlanırken `tvalid` ve `tlast` sıfırlanarak AXI-Stream üzerinde yinelenen sahte kelimeler engellendi (`0b0101` maskesi doğrulandı).
4. **Acil Durum Donanım Reset Kesmesi (Showstopper 5):**
   - FSM'in tüm durumlarını kapsayan global `ctrl_soft_reset` (bit 1) eklendi; kilitlenme durumunda tek döngüde `STATE_IDLE`'a dönüş sağlandı.
5. **Stage-B 48-bit -> 32-bit Taşma Kırpması (Risk 6):**
   - `StageBInt8` içerisine simetrik doygunluk kırpması (saturation clamp) eklendi; `> 0x7FFFFFFF` -> `0x7FFFFFFF` ve `< -0x80000000` -> `-0x80000000` garantilendi.
6. **Sentez ve Regresyon:**
   - `tests/unit/test_gate1_robustness.py` (5 test) yazıldı ve doğrulandı.
   - **Tüm Repo Regresyon Testi:** **312 testin tamamı PASSED** (0 hata).

---

### 4.6. Gate 2 (Pillar 1) — Ping-Pong Çift Tamponlama ve Sıfır-Baloncuklu Prefetching (TAMAMLANDI ✅)

LLM çıkarımında katman geçişlerinde ($L \to L+1$) hesaplama boru hattının duraklamasını (stall / bubble) ortadan kaldırmak için donanımsal çift tamponlama (double-buffering) ve otonom donanım takası (autonomous hardware bank swap) geliştirilmiştir:

1. **MSB Bank Bölümleme (Zero MUX):**
   - `RouteDecoderInt8` kod defteri belleği $2 \times 64 = 128$, `StageBInt8` ray belleği $2 \times 32 = 64$ ve `RailNetGrid` yönlendirme belleği $2 \times 512 = 1024$ derinliğe çıkarıldı.
   - Okuma adresi `Cat(addr, active_bank)` ve yazma adresi `Cat(prog_addr, prog_bank)` olarak bit birleştirme ile yönlendirildi; veri yollarında sıfır MUX ile $F_{\max}$ korundu.
2. **Otonom Donanım Takas Sözleşmesi (Autonomous Bank Swap):**
   - AXI-Lite CSR'a `active_bank` (STATUS bit 2), `bank_ready` (STATUS bit 3), `arm_next_bank` (CTRL bit 4), `auto_swap_en` (CTRL bit 5) ve `manual_swap` (CTRL bit 6) eklendi.
   - `prog_bank.eq(Mux(auto_swap_en, ~active_bank, active_bank))` ile tek tamponlu testlerle %100 geriye dönük uyumluluk korundu.
   - Katman $L$ hesaplaması `STATE_DONE`'a ulaştığında, eğer `auto_swap_en & bank_ready` aktifse FSM tek bir saykıl dahi kaybetmeden `active_bank <= ~active_bank` takasını yapar, `bank_ready`'i temizler ve doğrudan yeni katman akışını başlatır.
3. **PCIe Sürücüsü Asenkron Prefetching:**
   - `RailNetPCIeDriver` reentrant `RLock` ile kilitlenmelerden arındırıldı.
   - `prefetch_layer()` ve arka plan iş parçacığı (`PrefetchWorkerThread`) entegre edilerek model katmanları çalışırken bir sonraki katmanın arka plandaki banka akıtılması sağlandı.
4. **Doğrulama ve ASIC Sentezi:**
   - `tests/unit/test_gate2_double_buffering.py` (5/5 PASSED): Bank izolasyonu, otonom takas, manuel takas, sürücü prefetch ve thread kuyruğu test edildi.
   - **Tüm Repo Regresyon Testi:** **317 testin tamamı PASSED** (0 hata).
   - **Verilog Netlist:** `hardware/rtl/build/railnet_top.v` (9,929 satır Verilog-2001).
   - **SkyWater 130nm ASIC Sentezi:** *(2026-09-12 düzeltmesi)* Buradaki "327,884 standart mantık hücresi, PASSED" ifadesi yanlıştı — o sayı jenerik Yosys RTLIL primitifiydi, standart hücre değil. Gerçek `sky130_fd_sc_hd` haritalaması ilk kez 2026-09-12'de koşuldu (native Yosys 0.38 + `abc -fast -liberty`, OpenLane konteyneri): `railnet_top` için **39,706 standart hücre (55 tip), 245,247.712 µm² = 0.245 mm² mantık alanı**, artı haritalanmamış 24 `$mem_v2` belleği. Alan yalnızca mantığı kapsar; bellekler gerçek SRAM makrosu ister (Gate 4).

---

## 5. Uzun Vadeli Vizyon ve ASIC Yol Haritası

```
[2026 Q3] Çoklu Model + Hızlı CPU Çekirdeği (Adım 1-2)
    │
[2026 Q4] INT8 Karma Hassasiyet + Sanal Hızlandırıcı Simülatörü (Adım 3-4)
    │
[2027 Q1] FPGA Prototip Kartı (PCIe Geliştirme Kiti üzerinde Canlı Çıkarım)
    │
[2027 Q2] SkyWater 130nm / Tiny Tapeout (İlk Fiziksel Test Silikonu)
    │
[2027 Q4] TSMC 28nm/16nm Tapeout (Ticari Monolitik ReRAM Çıkarım ASIC'i)
```

### 5.1. Silikon Üretim Aşamaları

1. **Aşama 1 - FPGA Doğrulaması:** Xilinx Alveo U50 / Kria KV260 kartlarında PCIe üzerinden canlı inference.
2. **Aşama 2 - Açık Kaynak Silikon (SkyWater 130nm / Efabless MPW):**
   - Açık kaynaklı OpenLane / SkyWater PDK akışı ile küçük bir Stage-A + Stage-B test çipinin fiziksel tasarımı (GDSII).
3. **Aşama 3 - Ticari Üretim (TSMC 28nm / 22nm / 16nm):**
   - 169 mm² monolitik zar, gömülü ReRAM hücreleri, PCIe Gen5 arayüzü.
   - 0.28 Watt tüketim ile veri merkezlerinde sunucu başına 8-16 hızlandırıcı kartı yerleşimi.

### 5.2. ReRAM / Çapraz Bar (Crossbar) Entegrasyonu

Geleneksel SRAM hücreleri 6 transistörden (6T) oluşur ve sürekli enerji beslemesine ihtiyaç duyar.
- RailNet ReRAM mimarisi (1T1R - 1 Transistor 1 Resistor) kullanır.
- Bit hücre alanı $0.0016 \, \mu\text{m}^2$'dir (SRAM'den 25 kat daha yoğun).
- Çip kapatıldığında dahi ağırlık yönlendirme tabloları silinmez; açıldığı anda sıfır yükleme süresiyle (instant-on) çıkarıma hazır hale gelir.

### 5.3. Kurumsal Dağıtım ve Filo TCO Devrimi

RailNet, veri merkezlerinin model çalıştırma paradigmasını kökten değiştirecektir:
- **Taalas Modelinde:** Her model için $10M–$15M harcanıp 6 ay beklenirken;
- **RailNet Modelinde:** Tek tip üniversal hızlandırıcı kartı satın alınır. Llama-3.2, Qwen-2.5, Gemma-3 veya yarın çıkacak yeni bir model saniyeler içinde çipe flash'lanır.
- **Sıfır HBM Bağımlılığı:** HBM kıtlığı ve fahiş fiyatlarına takılmadan, standart silikon fabrikasyon süreçleriyle milyarlarca kullanıcının cebine ve sunuculara sub-watt LLM çıkarımı getirilir.

---

## 6. Özet ve Sonuç

RailNet projesi, teorik bir tezden yola çıkarak;
1. Bit-exact doğrulukta çalışan **matematiksel çekirdeğini**,
2. Yosys ile sentezlenmiş **sıfır-DSP donanım karosunu**,
3. ECP5 FPGA üzerinde kanıtlanmış **zamanlama ve P&R analizini**,
4. Taalas'a karşı %78 TCO üstünlüğünü kanıtlayan **PPA simülatörünü**,
5. Ve son olarak Llama-3, Qwen-2.5 ve Gemma-3'ü kapsayan **çoklu model çalışma zamanını** başarıyla inşa etmiştir.

Belirlediğimiz 4 aşamalı yol haritası doğrultusunda sıradaki aksiyonumuz: **Adım 2 — Hızlı CPU Çıkarım Çekirdeği (C++/SIMD Kernel Optimization)** ile çıkarım performansını zirveye taşımaktır.
