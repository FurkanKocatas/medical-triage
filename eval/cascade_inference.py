#!/usr/bin/env python3
"""
cascade_inference.py — 3-layer inference cascade.

L1 (SmolLM2-360M): fast binary emergency filter
  - if triaj == "Yesil": return immediately (low risk)
  - else: escalate to L2

L2 (Qwen2.5-1.5B): full triaj + bolum classification
  - returns {triaj, bolum}
  - if triaj != "Yesil": escalate to L3 for reasoning

L3 (Gemma-4-E2B): reasoning + final verification
  - validates L2 output, provides `neden`
  - can upgrade triaj if critical phrases detected

All outputs get a disclaimer injected.

Usage:
  python eval/cascade_inference.py --input "Ani gogus agrisi, sol kolum uyusuyor"
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Desktop/medical-triage/.hf_cache"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DISCLAIMER = (
    "Bu sistem sadece bir on yonlendirme aracidir, tibbi tani koymaz. "
    "Acil durumlarda 112'yi arayin."
)

INSTRUCTION_L12 = (
    "Asagidaki hasta semptom metnini analiz et. Triaj seviyesini "
    "(Yesil/Sari/Kirmizi) ve en uygun tibbi bolumu belirle. "
    "Cevabini SADECE gecerli bir JSON objesi olarak ver."
)

INSTRUCTION_L3 = (
    "Asagidaki hasta semptomu ve on-analiz cikti icin tibbi gerekce (neden) yaz. "
    "Gerekirse triaj seviyesini duzelt. JSON formatinda cevap ver."
)


def format_prompt_l12(text: str) -> str:
    return (f"### Gorev\n{INSTRUCTION_L12}\n\n"
            f"### Semptom\n{text.strip()}\n\n### Cevap\n")


def format_prompt_l3(text: str, prev: dict) -> str:
    return (f"### Gorev\n{INSTRUCTION_L3}\n\n"
            f"### Semptom\n{text.strip()}\n\n"
            f"### On-Analiz\n{json.dumps(prev, ensure_ascii=False)}\n\n"
            f"### Cevap\n")


def parse_json(txt: str) -> dict:
    m = re.search(r"\{[^{}]*\}", txt, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def load(model_id: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter and Path(adapter).exists():
        m = PeftModel.from_pretrained(m, adapter)
        m = m.merge_and_unload()
    m.eval()
    return tok, m


def generate(tok, model, prompt: str, max_new=120) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, temperature=1.0, top_p=1.0,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class Cascade:
    def __init__(self, l1_adapter: str | None, l2_adapter: str | None, l3_adapter: str | None):
        print("[*] loading L1 (SmolLM2)...")
        self.l1_tok, self.l1 = load("HuggingFaceTB/SmolLM2-360M", l1_adapter)
        print("[*] loading L2 (Qwen2.5)...")
        self.l2_tok, self.l2 = load("Qwen/Qwen2.5-1.5B", l2_adapter)
        print("[*] loading L3 (Gemma-4)...")
        self.l3_tok, self.l3 = load("google/gemma-4-E2B", l3_adapter)

    def __call__(self, text: str) -> dict:
        t0 = time.time()
        r1 = parse_json(generate(self.l1_tok, self.l1, format_prompt_l12(text), max_new=80))
        t1 = time.time()
        triaj_l1 = (r1.get("triaj") or "").strip().capitalize()

        if triaj_l1 == "Yesil":
            return {
                "triaj": "Yesil",
                "bolum": r1.get("bolum", "Bilinmiyor"),
                "neden": "L1 hizli filtre: dusuk aciliyet",
                "layers_used": ["L1"],
                "latency_sec": round(t1 - t0, 3),
                "disclaimer": DISCLAIMER,
            }

        r2 = parse_json(generate(self.l2_tok, self.l2, format_prompt_l12(text), max_new=120))
        t2 = time.time()
        triaj_l2 = (r2.get("triaj") or "").strip().capitalize()
        bolum_l2 = (r2.get("bolum") or "").strip()

        if triaj_l2 == "Yesil":
            return {
                "triaj": "Yesil",
                "bolum": bolum_l2 or r1.get("bolum", "Bilinmiyor"),
                "neden": (r2.get("neden") or "L2 siniflandirici: dusuk aciliyet").strip(),
                "layers_used": ["L1", "L2"],
                "latency_sec": round(t2 - t0, 3),
                "disclaimer": DISCLAIMER,
            }

        prev = {"triaj": triaj_l2 or triaj_l1, "bolum": bolum_l2 or r1.get("bolum", "")}
        r3 = parse_json(generate(self.l3_tok, self.l3, format_prompt_l3(text, prev), max_new=150))
        t3 = time.time()
        final_triaj = (r3.get("triaj") or triaj_l2 or triaj_l1 or "Sari").strip().capitalize()
        final_bolum = (r3.get("bolum") or bolum_l2 or "Bilinmiyor").strip()
        final_neden = (r3.get("neden") or "").strip() or "L3 reasoning gerekce yok"

        return {
            "triaj": final_triaj,
            "bolum": final_bolum,
            "neden": final_neden,
            "layers_used": ["L1", "L2", "L3"],
            "latency_sec": round(t3 - t0, 3),
            "disclaimer": DISCLAIMER,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--l1_adapter", default="models/smollm2_360m_qlora/final_adapter")
    ap.add_argument("--l2_adapter", default="models/qwen25_15b_qlora/final_adapter")
    ap.add_argument("--l3_adapter", default="models/gemma4_e2b_qlora/final_adapter")
    args = ap.parse_args()

    cascade = Cascade(args.l1_adapter, args.l2_adapter, args.l3_adapter)
    result = cascade(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
