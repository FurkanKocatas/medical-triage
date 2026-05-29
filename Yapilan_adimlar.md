# Yapilan Adimlar — Tibbi Triaj ve Bolum Yonlendirme Projesi

**Proje:** NVIDIA DGX GB10 (Blackwell) uzerinde Turkce LLM tabanli tibbi triaj + bolum siniflandirici
**Host:** nvidia@10.199.0.10 (spark-25ed, aarch64, Ubuntu 24.04)
**GPU:** NVIDIA GB10 (Grace-Blackwell), BF16/FP8 destekli
**Python:** 3.12.3

## Modeller
- **L1 (Hizli filtre):** HuggingFaceTB/SmolLM2-360M
- **L2 (Ana siniflandirici):** Qwen/Qwen2.5-1.5B
- **L3 (Reasoning):** google/gemma-4-E2B

## Veri Setleri
- HF: `alibayram/doktorsitesi` (5k sample)
- HF: `alibayram/turkish-hospital-medical-articles` (5k sample)
- Kaggle: `doguseryarar/turkish-medical-emergency-triage-dataset` (archive.zip, supervised label kaynagi)
- **Toplam:** 10k HF sample + Kaggle triaj kural/etiket kaynagi

## Hedefler
1. Multi-task: triaj (Yesil/Sari/Kirmizi) + bolum (Dahiliye/Kardiyoloji/...)
2. QLoRA + LoRA karsilastirma (bf16, gradient_checkpointing)
3. Metrikler: Accuracy, Macro-F1, Weighted-F1, **False Green Rate**, inference ms, model MB, GPU mem
4. Zero-shot vs fine-tuned karsilastirma
5. Cascade inference: L1 → L2 → L3
6. JSON output: `{triaj, bolum, neden}` + disclaimer

---

## Adim Log

### Adim 1: SSH + Ortam Dogrulama (2026-04-24)
- SSH key push: paramiko ile local `id_ed25519.pub` → `nvidia@10.199.0.10:~/.ssh/authorized_keys`
- `~/.ssh/config` local `sparknode` alias eklendi
- Dogrulama:
  - Host: `spark-25ed`
  - Kernel: `Linux 6.17.0-1014-nvidia aarch64`
  - GPU: `NVIDIA GB10` (1 GPU)
  - Python: `3.12.3`
  - Disk: 3.0T free

### Adim 2: Proje Dizin Yapisi
Olusturulan dizinler:
\`\`\`
~/Desktop/medical-triage/
├── configs/
├── data/
│   ├── raw/        # ham datasetler (Kaggle zip, HF cache)
│   ├── sampled/    # 5k+5k rastgele sample
│   └── labeled/    # triaj + bolum etiketli JSONL
├── eval/           # degerlendirme scriptleri
├── logs/           # egitim loglari
├── models/         # indirilen + fine-tuned modeller
├── notebooks/      # analiz scriptleri
├── scripts/        # preprocessing, sampling, labeling
└── train/          # QLoRA/LoRA egitim scriptleri
\`\`\`
Git init yapildi.


### Adim 3: Kaggle Veri Transferi
- SCP: `C:\Users\meru\Downloads\archive.zip` → `data/raw/kaggle_triage.zip`
- Unzip: `medical_data.json` (920 ornek, 11770 satir)
- Sema: `{id, input_text, symptoms, urgency_level (1-5), urgency_label, response, reasoning}`
- Triaj etiket dagilimi (urgency_label):
  - NORMAL: 309 → **Yesil**
  - ACIL/ACIL: 418 → **Sari**
  - COK ACIL: 193 → **Kirmizi**

### Adim 4: Python Sanal Ortam + Bagimliliklar (aarch64)
- `~/Desktop/medical-triage/.venv` olusturuldu (Python 3.12.3)
- Torch: `2.11.0+cu128` (PyTorch CU128 aarch64 wheel)
- GPU dogrulama: GB10 bf16 OK, compute capability (12, 1) = Blackwell
- HF stack: transformers 5.6.2, peft 0.19.1, trl 1.2.0, datasets 4.8.4, accelerate 1.13.0
- QLoRA: bitsandbytes 0.49.2 aarch64 wheel, 4bit NF4 linear testi OK
- Diger: numpy 2.4, pandas 3.0, scikit-learn 1.8, sentencepiece, protobuf, evaluate

### Adim 5: HF Dataset Indirme
- Token: ~/.cache/huggingface/token (login: Meruthewarden)
- HF_HOME: proje-local `.hf_cache/` (default `/root` permission problemli)
- **alibayram/doktorsitesi** (gated, access OK):
  - 150105 train, 37527 test, 187632 title_spec_combined, 8000+800 balanced
  - Kolonlar: `doctor_title, doctor_speciality, question_content, question_answer`
  - 116 alt-uzmanlik alani
- **alibayram/turkish-hospital-medical-articles** (gated, BLOCKED):
  - User HF hesabi access talebi gerekiyor
  - Simdilik atlaniyor, onay sonrasi 5k eklenecek

### Adim 6: Doktorsitesi 5k Random Sampling
- Script: `scripts/sample_data.py` (seed=42, utf-8, min 20 karakter, Diger elenir)
- 116 alt-uzmanligi 23 ana bolume normalize (regex slug match)
- Cikti: `data/sampled/doktorsitesi_5k.jsonl` (5000 satir)
- Bolum dagilimi: Kadin Dogum 1374, Cocuk Sagligi 513, Dahiliye 501, Noroloji 492, Uroloji 370, Ortopedi 279, Genel Cerrahi 254, Plastik 148, KBB 132, Kardiyoloji 122, Fizik Tedavi 122, Psikiyatri 116, Dis 113, Dermatoloji 96, Psikoloji 85, Goz 66, Gogus 52, Beslenme 44, Anestezi 43, Enfeksiyon 38, Radyoloji 29, Aile Hek 6, Acil Tip 5

### Adim 7: Kaggle Triaj Kural Extraction
- Script: `scripts/extract_triage_rules.py`
- Her triaj sinifi icin discriminative keyword listeleri (P(word|class) / P(word|not-class))
- En guclu Kirmizi keywordler: ani siddetli bas agrisi, ani siddetli karin agrisi, bilinc bulanikligi, ense sertligi, ani gorme kaybi, afazi, hematemez, sok, yuzde uyusma
- Kritik fraz listesi (STRONG_RED_PATTERNS): kalp krizi gecir*, inme gecir, ani yuzde felc, ani siddetli bas agri, ensede sert, havale gecir, konvulziy, dudaklar morar, siyanoz, anafilaktik sok, septik sok, ani kan gelm
- Akut baslangic markeri (ACUTE_MARKERS): ani*, aniden, birdenbire, cok siddetli, acil servis, ambulans, 112
- Gecmis zaman filtresi (PAST_TENSE_MARKERS): gecirdim/gecirmis, olmustu, yil(lar) once, gecmiste
- Cikti: `configs/triage_rules.json`

### Adim 8: Labeler Uygulamasi
- Script: `scripts/label_data.py`
- Oncelik:
  1. STRONG_RED_PATTERNS regex match + past-tense filtresi -> Kirmizi
  2. Kritik belirti + akut baslangic markeri + gecmis zaman filtresi -> Kirmizi
  3. Hospital articles (varsa) -> Yesil (info metni)
  4. Skor bazli argmax (Kirmizi/Sari/Yesil keywords agirlikli)
  5. Mild phrase + dusuk Kirmizi skor -> Yesil
  6. Default -> Sari (hasta sorusu baseline)
- Cikti: `data/labeled/labeled_all.jsonl` (5000 satir)
- Triaj dagilimi: Sari 3262 (%65), Yesil 1527 (%31), Kirmizi 211 (%4)
- Realistik: doktorsitesi patient-question base-rate ile uyumlu (gercek acil vakalar 112 cagiriyor)

### Adim 9: 3 Model Indirme (HF snapshot)
- `HuggingFaceTB/SmolLM2-360M` (362M params, 724MB)
- `Qwen/Qwen2.5-1.5B` (1.5B params, 3.1GB)
- `google/gemma-4-E2B` (5.1B params, 10.2GB expanded - E2B edge compressed)
- Hepsi bf16 load OK, zero-shot Turkce generation dogrulandi
- Gemma-4 zero-shot test: "Ani gogus agrisi + kol uyusmasi" -> "Triaj seviyesi 1" (dogru, acute MI)

### Adim 10: Train/Val/Test Split + SFT Formatlama
- Script: `scripts/split_and_format.py`
- 80/10/10 stratified split (triaj+bolum pair stratify), seed=42
- Train: 3972, Val: 475, Test: 553
- SFT prompt template: "### Gorev\n{INST}\n### Semptom\n{text}\n### Cevap\n" + JSON response
- Cikti: `data/labeled/{train,val,test}.jsonl` (prompt/response/full columns)

### Adim 11: QLoRA Training Script
- Script: `train/qlora_train.py` (tek script, config-driven)
- 4bit nf4 + bf16 compute, paged_adamw_8bit, gradient_checkpointing (reentrant=False)
- TRL 1.2 SFTTrainer: select_columns(["full"]).rename_column("full", "text"), dataset_text_field="text"
- pick_targets(): gemma -> "all-linear", diger -> [q/k/v/o/gate/up/down]_proj
- LoRA r=16, alpha=32, dropout=0.05
- Configs: smollm2 (bs=8 ga=2 lr=3e-4 ep=3), qwen25 (bs=4 ga=4 lr=2e-4 ep=2), gemma4 (bs=2 ga=8 lr=1e-4 ep=2), qwen25_lora (same qwen25 ama use_4bit=false karsilastirma)

### Adim 12: SmolLM2-360M QLoRA Training
- Komut: HF_HOME=... .venv/bin/python train/qlora_train.py --config configs/smollm2.json
- Sure: 62 dk, 3 epoch, 747 step
- eval_loss: 1.625, train_loss: 1.796, eval_mean_token_accuracy: 0.6636
- Adapter: models/smollm2_360m_qlora/final_adapter/ (17MB)

### Adim 13: Qwen2.5-1.5B QLoRA Training
- Komut: HF_HOME=... .venv/bin/python train/qlora_train.py --config configs/qwen25.json
- Sure: 89 dk, 2 epoch, 498 step
- eval_loss: 1.662, train_loss: 1.796, eval_mean_token_accuracy: 0.6812
- Adapter: models/qwen25_15b_qlora/final_adapter/ (37MB)

### Adim 14: Eval + Cascade + Report Scriptleri
- `eval/evaluate.py`: accuracy, macro-F1, weighted-F1, FGR, FNR-Red, latency, size
- `eval/cascade_inference.py`: L1->L2->L3 cascade + disclaimer
- `eval/run_all_evals.sh`: tum modeller icin zero-shot + FT eval bash runner
- `scripts/generate_report.py`: eval/results/*.json -> REPORT.md karsilastirma tablosu

### Adim 15: DGX Spark Erisim Kesintisi + Local Transfer
- Spark erisimi 2 gunluk kesinti geldi, Gemma-4 training yapilmadi
- tar | ssh | tar stream ile proje C:\Users\meru\Desktop\LLMOdev\medical-triage\ altinda yerele alindi
- .hf_cache (27GB, base modeller) ve .venv atlandi (yeniden indirilebilir)
- RESUME_HERE.md olusturuldu: Gemma-4 training, hospital-articles sample, LoRA comparison, eval, REPORT adimlari dokumentelendi
