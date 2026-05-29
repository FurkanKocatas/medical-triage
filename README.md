# Türkçe Tıbbi Triaj ve Bölüm Yönlendirme için Küçük Dil Modellerinin İnce Ayarı

Türkçe serbest metin hasta semptomlarından **(i) triaj aciliyet seviyesi** (Yeşil / Sarı / Kırmızı) ve
**(ii) hastane bölüm yönlendirmesi**'ni birlikte üreten, parametre verimli ince ayar (QLoRA / LoRA) ile
eğitilmiş **Küçük Dil Modeli (SLM)** karşılaştırmalı benchmark çalışması.

Zorunlu model setinden **SmolLM2-360M**, **Qwen2.5-1.5B** ve **Gemma-4-E2B** modelleri değerlendirilmiştir.

> ⚠️ **Sorumluluk reddi:** Bu bir **araştırma prototipidir**. Klinik karar verme aracı **değildir**.
> Kırmızı sınıfı için yanlış-negatif oranı (FNR-Red = 0,529) klinik kabul eşiğinin (literatürde < %5–10)
> oldukça üzerindedir. Yetkin hekim doğrulaması olmadan kullanılmamalıdır.

---

## Ana Sonuçlar (Standart Sonuç Tablosu)

| Model | Parametre | FT Yöntemi | Accuracy | Macro-F1 | Eğitim Süresi | Inference (ms/örnek) | Model Boyutu (MB) |
|---|---|---|---|---|---|---|---|
| SmolLM2-360M | 362 M | QLoRA r=16 | 0,562 | 0,244 | 62,3 dk | 1.706 | 21,0 (adapter) |
| Qwen2.5-1.5B | 1,5 B | QLoRA r=16 | 0,622 | 0,260 | 89,2 dk | 1.391 | 48,4 (adapter) |
| Qwen2.5-1.5B | 1,5 B | LoRA bf16 r=16 | 0,624 | 0,256 | 35,6 dk | 1.463 | 85,3 (adapter) |
| Gemma-4-E2B | ~2 B | QLoRA r=8 | 0,624 | 0,256 | 42,8 dk | 2.733 | 70,3 (adapter) |
| **Qwen2.5-1.5B (dengeli)** | 1,5 B | LoRA bf16 r=16 | 0,618 | **0,390** | ~36 dk | 1.407 | 85,3 (adapter) |

- **En iyi yapılandırma:** Kırmızı sınıfı 4× oversample edilmiş **dengeli Qwen2.5-1.5B** — Macro-F1 0,256 → 0,390 (+%52 göreli), FNR-Red 1,000 → 0,529.
- Tüm **zero-shot** yapılandırmaları geçerli JSON/Türkçe etiket üretemedi (accuracy ≈ 0).
- L1→L2→L3 **cascade** mimarisi tek başına dengeli Qwen ile aynı sonucu verdi, ~11× ek gecikme getirdi.

Tam analiz, grafikler ve hata analizi için final raporuna bakınız: [`rapor1.pdf`](rapor1.pdf).

---

## Repo Yapısı

```
medical-triage/
├── configs/              # Model + eğitim yapılandırmaları (JSON) + triage_rules.json
├── data/
│   ├── raw/              # Kaggle triaj kaynağı (kural/etiket çıkarımı için)
│   ├── sampled/          # doktorsitesi 5k örnek
│   └── labeled/          # train / val / test (.jsonl) + istatistikler
├── scripts/              # Veri işleme: sample, rule-extract, label, split, plot, report
├── train/qlora_train.py  # Tek, config-driven QLoRA/LoRA eğitim scripti
├── eval/                 # evaluate.py, cascade_*.py, run_all_evals.sh, results/, plots/
├── models/               # Fine-tune adapter'ları (final_adapter/)
├── requirements.txt
├── rapor1.pdf            # Final raporu
└── README.md
```

---

## Kurulum

**Önkoşullar:** Python 3.12, CUDA 12.x destekli NVIDIA GPU (eğitim için; QLoRA ~12 GB VRAM'e sığar).

```bash
# 1) Sanal ortam
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) PyTorch'u platformunuza uygun CUDA wheel'i ile kurun
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3) Kalan bağımlılıklar
pip install -r requirements.txt

# 4) Hugging Face girişi (doktorsitesi GATED bir veri kümesidir — erişim onayı gerekir)
huggingface-cli login
```

> **Not:** Tüm komutlarda HF önbelleği proje-local tutulur: `export HF_HOME="$PWD/.hf_cache"`
> (Windows PowerShell: `$env:HF_HOME = "$PWD\.hf_cache"`).

---

## Çalıştırma

### 1) Veri hazırlama (yeniden üretmek için)

```bash
export HF_HOME="$PWD/.hf_cache"
python scripts/sample_data.py --dataset doktorsitesi --n 5000 --seed 42
python scripts/extract_triage_rules.py
python scripts/label_data.py
python scripts/split_and_format.py --seed 42        # 80/10/10 stratified
```

> Etiketli `train/val/test.jsonl` repoda hazır gelir; yukarıdaki adımlar yalnızca sıfırdan üretim içindir.

### 2) Eğitim (config-driven)

```bash
export HF_HOME="$PWD/.hf_cache"
python train/qlora_train.py --config configs/smollm2.json
python train/qlora_train.py --config configs/qwen25.json
python train/qlora_train.py --config configs/qwen25_lora.json            # LoRA bf16 (karşılaştırma)
python train/qlora_train.py --config configs/qwen25_lora_balanced.json   # dengeli (en iyi)
python train/qlora_train.py --config configs/gemma4.json
```

### 3) Değerlendirme

```bash
# Zero-shot + fine-tuned, 3 model için tek seferde:
bash eval/run_all_evals.sh

# Tek model örneği:
python eval/evaluate.py --model_id Qwen/Qwen2.5-1.5B \
  --adapter models/qwen25_15b_lora_balanced/final_adapter --tag qwen25_lora_balanced

# Cascade (L1→L2→L3):
python eval/cascade_eval.py --tag cascade_balanced_v2
```

### 4) Grafik ve rapor tablosu

```bash
python scripts/plot_results.py        # eval/plots/*.png
python scripts/generate_report.py     # REPORT.md karşılaştırma tablosu
```

---

## Modeller

| Katman | Base Model (HF) | Adapter |
|---|---|---|
| L1 (hızlı filtre) | `HuggingFaceTB/SmolLM2-360M` | `models/smollm2_360m_qlora/final_adapter` |
| L2 (ana sınıflandırıcı) | `Qwen/Qwen2.5-1.5B` | `models/qwen25_15b_lora_balanced/final_adapter` |
| L3 (reasoning) | `google/gemma-4-E2B` | `models/gemma4_e2b_qlora_balanced/final_adapter` |

Adapter'lar repoda gelir (`final_adapter/`). Eğitim checkpoint'leri (optimizer state) `.gitignore` ile hariç tutulmuştur.

---

## Tekrarlanabilirlik

- **Sabit tohum:** Tüm örnekleme, bölme ve eğitim adımları `seed=42` ile yapılmıştır.
- **Veri bölme:** 80/10/10 (train 3.972 / val 475 / test 553), triaj×bölüm çiftine göre stratifiye.
- **Bilinen sınırlama:** Deneyler tek tohum (seed=42) ile koşulmuştur; çoklu tohum (≥3) ile ortalama ± standart sapma
  ve istatistiksel anlamlılık testleri **gelecek çalışmaya** bırakılmıştır (bkz. rapor, Bölüm 8–9).

---

## Veri Setleri ve Lisans

| Kaynak | Kullanım | Erişim |
|---|---|---|
| `alibayram/doktorsitesi` (HF) | 5k örnek (semptom→bölüm) | **Gated** — HF üzerinden erişim onayı gerekir |
| Kaggle Türkçe Tıbbi Acil Triaj | Triaj kural/etiket kaynağı | Kaggle kullanım koşullarına tabidir |

Türetilmiş veri dosyaları yalnızca bu akademik proje kapsamında, kaynak veri kümelerinin lisans koşulları
çerçevesinde paylaşılmıştır. Kaynak verileri kendi hesabınızla edinmeniz önerilir.

---

## Yazarlar

- **Ebubekir Bayar** — 258273002001
- **Furkan Kocataş** — 258273002008

Selçuk Üniversitesi, Fen Bilimleri Enstitüsü, Bilgisayar Mühendisliği (Yüksek Lisans) — Büyük Dil Modelleri Dersi, 2026.
