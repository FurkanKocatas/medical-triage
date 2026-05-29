# RESUME_HERE — Tibbi Triaj Projesi

**Son durum (2026-04-24):** DGX Spark erisimi kesildi. Proje `C:\Users\meru\Desktop\LLMOdev\medical-triage\` altinda yerelde.

---

## Tamamlanan

1. SSH + Spark env (aarch64 Ubuntu 24.04, CUDA 13, PyTorch 2.11 cu128, bnb 0.49.2)
2. Veri:
   - `alibayram/doktorsitesi` 5k sample → `data/sampled/doktorsitesi_5k.jsonl`
   - Kaggle `archive.zip` → triaj kural kaynagi
   - Etiketli veri: `data/labeled/{train,val,test}.jsonl` (3972/475/553, stratified, seed=42)
   - Dagilim: Sari %65, Yesil %31, Kirmizi %4
3. Modeller (base):
   - SmolLM2-360M (362M / 724MB)
   - Qwen2.5-1.5B (1.5B / 3097MB)
   - Gemma-4-E2B (5.1B / 10219MB)
4. QLoRA Fine-tune ADAPTERS:
   - **SmolLM2**: `models/smollm2_360m_qlora/final_adapter/` (17MB, eval_loss 1.625, 62 dk)
   - **Qwen2.5**: `models/qwen25_15b_qlora/final_adapter/` (37MB, eval_loss 1.662, 89 dk)
   - **Gemma-4**: YAPILMADI (spark erisimi kesildi)
5. Tum scriptler: sample_data, extract_triage_rules, label_data, split_and_format, verify_models, qlora_train, evaluate, cascade_inference, generate_report
6. Configs: smollm2.json, qwen25.json, gemma4.json, qwen25_lora.json

## Yapilmayanlar (devam edilecek)

- **Gemma-4-E2B QLoRA training** — config hazir (`configs/gemma4.json`: bs=2 ga=8 lr=1e-4 epochs=2 max_len=896 target="all-linear"). Komut:
  ```bash
  HF_HOME=$PWD/.hf_cache .venv/bin/python train/qlora_train.py --config configs/gemma4.json
  ```
- **hospital-articles HF dataset** — gated, erisim iznini beklerken kesildi. https://huggingface.co/datasets/alibayram/turkish-hospital-medical-articles → "Agree and access"
- 5k hospital sample + yeniden etiketleme (10k kombine)
- **LoRA (no 4bit) Qwen** — `configs/qwen25_lora.json` hazir, karsilastirma icin
- **Zero-shot + FT eval** — `eval/run_all_evals.sh` hazir, tum modeller icin
- **REPORT.md** — `scripts/generate_report.py` ile uretilir

## Yeniden baslama (Spark'a eris tekrar gelince)

1. Projeyi spark'a geri at:
   ```bash
   scp -r /c/Users/meru/Desktop/LLMOdev/medical-triage sparknode:~/Desktop/
   ssh sparknode 'cd ~/Desktop/medical-triage && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
   ```
   Not: `requirements.txt` yok — paketler: `torch==2.11.0+cu128` (aarch64), transformers, peft, trl, bitsandbytes==0.49.2, datasets, accelerate, huggingface_hub, sklearn, pandas.

2. Base modelleri yeniden indir:
   ```bash
   HF_HOME=$PWD/.hf_cache .venv/bin/python scripts/verify_models.py
   ```

3. Gemma-4 QLoRA baslat:
   ```bash
   HF_HOME=$PWD/.hf_cache .venv/bin/python train/qlora_train.py --config configs/gemma4.json > logs/gemma4_train.log 2>&1 &
   ```

4. hospital-articles erisim alindiysa:
   ```bash
   HF_HOME=$PWD/.hf_cache .venv/bin/python scripts/sample_data.py --dataset hospital
   HF_HOME=$PWD/.hf_cache .venv/bin/python scripts/label_data.py  # yeniden etiketle
   .venv/bin/python scripts/split_and_format.py  # yeniden split
   ```
   Sonra SmolLM2 + Qwen2.5 FT tekrar baslatilmali (yeni train set icin).

5. LoRA (no 4bit) karsilastirma:
   ```bash
   HF_HOME=$PWD/.hf_cache .venv/bin/python train/qlora_train.py --config configs/qwen25_lora.json
   ```

6. Eval:
   ```bash
   bash eval/run_all_evals.sh
   ```

7. Rapor:
   ```bash
   .venv/bin/python scripts/generate_report.py
   ```

## Dosya Yapisi (transfer sonrasi)

```
LLMOdev/medical-triage/
├── Yapilan_adimlar.md        # tum adimlar log
├── RESUME_HERE.md            # bu dosya
├── configs/                  # 4 model config
├── data/
│   ├── sampled/              # 5k doktorsitesi
│   ├── raw/                  # Kaggle triaj
│   └── labeled/              # train/val/test jsonl
├── models/
│   ├── smollm2_360m_qlora/final_adapter/   # FT adapter
│   └── qwen25_15b_qlora/final_adapter/     # FT adapter
├── scripts/                  # tum data/rule/label/split scriptleri
├── train/qlora_train.py      # tek script, config-driven
├── eval/                     # evaluate + cascade + run_all
└── logs/                     # training loglari
```

## Onemli Notlar

- **HF_HOME** her komutta `$PWD/.hf_cache` olmali (root cache yazma iznimiz yok).
- **bfloat16** default, Blackwell (compute 12.1) native destek.
- **TRL 1.2 API**: `dataset_text_field="text"`, `select_columns(["full"]).rename_column("full", "text")` sart.
- **Kirmizi orani dusuk (%4)**: STRONG_RED_PATTERNS + ACUTE_MARKERS + PAST_TENSE filtre sonrasi normal. Eval sirasinda FGR/FNR-Red dikkatli bak.
- **Gemma-4 target_modules**: `"all-linear"` (diger modellerde list halinde spesifik).
- **hf token**: `~/.cache/huggingface/token` (Meruthewarden login) — gerekirse tekrar `huggingface-cli login`.

## Metrics (mevcut adapter'lar)

| Model | eval_loss | train_loss | Sure | Adapter MB |
|---|---|---|---|---|
| SmolLM2-360M | 1.625 | 1.796 | 62 dk | 17 |
| Qwen2.5-1.5B | 1.662 | 1.796 | 89 dk | 37 |
| Gemma-4-E2B | — | — | — | — |
