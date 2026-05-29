#!/usr/bin/env python3
"""
qlora_train.py — Unified QLoRA fine-tuning for SmolLM2 / Qwen2.5 / Gemma-4.

Usage:
  python train/qlora_train.py --config configs/smollm2.json
  python train/qlora_train.py --config configs/qwen25.json
  python train/qlora_train.py --config configs/gemma4.json

Config JSON fields:
  model_id       : HF repo id
  output_dir     : where to save adapter
  lora_r         : LoRA rank (default 16)
  lora_alpha     : LoRA alpha (default 32)
  lora_dropout   : (default 0.05)
  target_modules : list of module names (or "all-linear")
  use_4bit       : bool, default true (QLoRA)
  epochs         : default 2
  batch_size     : per-device train batch (default 2)
  grad_accum     : gradient accumulation steps (default 8)
  lr             : learning rate (default 2e-4)
  max_length     : token cap (default 1024)
  seed           : (default 42)
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Desktop/medical-triage/.hf_cache"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


DEFAULT_TARGETS = {
    "smollm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": "all-linear",
}


def pick_targets(model_id: str, cfg_targets):
    if cfg_targets:
        return cfg_targets
    mid = model_id.lower()
    if "gemma" in mid:
        return DEFAULT_TARGETS["gemma"]
    if "qwen" in mid:
        return DEFAULT_TARGETS["qwen"]
    return DEFAULT_TARGETS["smollm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    model_id = cfg["model_id"]
    output_dir = cfg["output_dir"]
    epochs = cfg.get("epochs", 2)
    bs = cfg.get("batch_size", 2)
    ga = cfg.get("grad_accum", 8)
    lr = cfg.get("lr", 2e-4)
    max_len = cfg.get("max_length", 1024)
    seed = cfg.get("seed", 42)
    packing = cfg.get("packing", False)
    use_4bit = cfg.get("use_4bit", True)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    lora_dropout = cfg.get("lora_dropout", 0.05)
    targets = pick_targets(model_id, cfg.get("target_modules"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[*] Model: {model_id}")
    print(f"[*] Output: {output_dir}")
    print(f"[*] Targets: {targets}")
    print(f"[*] 4bit: {use_4bit} | r={lora_r} alpha={lora_alpha} drop={lora_dropout}")
    print(f"[*] bs={bs} ga={ga} lr={lr} epochs={epochs} max_len={max_len}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_file = cfg.get("train_file", "data/labeled/train.jsonl")
    train_ds = load_dataset("json", data_files=train_file, split="train")
    val_ds = load_dataset("json", data_files="data/labeled/val.jsonl", split="train")
    print(f"[*] train_file: {train_file} (n={len(train_ds)})")
    keep_cols = ["full"]
    train_ds = train_ds.select_columns(keep_cols).rename_column("full", "text")
    val_ds = val_ds.select_columns(keep_cols).rename_column("full", "text")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=ga,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        fp16=False,
        tf32=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=max_len,
        packing=packing,
        dataset_text_field="text",
        report_to="none",
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
    )

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    adapter_path = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tok.save_pretrained(str(adapter_path))

    metrics = {
        "model_id": model_id,
        "training_time_sec": dt,
        "training_time_min": round(dt / 60, 2),
        "epochs": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "config": cfg,
    }
    with (Path(output_dir) / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[+] training done in {dt/60:.1f} min")
    print(f"[+] adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
