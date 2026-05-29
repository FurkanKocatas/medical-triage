#!/usr/bin/env python3
"""
evaluate.py — Evaluate a model (base or fine-tuned adapter) on test set.

Computes: Accuracy, Macro-F1, Weighted-F1 (triaj + bolum),
          False Green Rate (FGR), inference ms/sample,
          model size MB, peak GPU memory MB.

Usage:
  python eval/evaluate.py --model_id Qwen/Qwen2.5-1.5B --tag qwen25_zeroshot
  python eval/evaluate.py --model_id Qwen/Qwen2.5-1.5B --adapter models/qwen25_15b_qlora/final_adapter --tag qwen25_ft
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/Desktop/medical-triage/.hf_cache"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from peft import PeftModel

TRIAJ_CLASSES = ["Yesil", "Sari", "Kirmizi"]


def parse_output(text: str) -> dict:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {"triaj": None, "bolum": None, "neden": None, "parse_ok": False}
    try:
        obj = json.loads(m.group(0))
        triaj = (obj.get("triaj") or "").strip()
        bolum = (obj.get("bolum") or "").strip()
        neden = (obj.get("neden") or "").strip()
        if triaj.lower() in ["yeşil", "yesil", "green"]: triaj = "Yesil"
        elif triaj.lower() in ["sarı", "sari", "yellow"]: triaj = "Sari"
        elif triaj.lower() in ["kırmızı", "kirmizi", "red"]: triaj = "Kirmizi"
        return {"triaj": triaj, "bolum": bolum, "neden": neden, "parse_ok": True}
    except Exception:
        return {"triaj": None, "bolum": None, "neden": None, "parse_ok": False}


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test", default="data/labeled/test.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="eval only first N samples (debug)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    print(f"[*] device={DEVICE} dtype={DTYPE}")

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=DTYPE, device_map=DEVICE,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[*] adapter loaded + merged: {args.adapter}")
    model.eval()
    t_load = time.time() - t0

    # model size on disk
    if args.adapter:
        size_mb = dir_size_mb(args.adapter)
        size_desc = "adapter only"
    else:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(args.model_id, cache_dir=os.environ["HF_HOME"])
        size_mb = dir_size_mb(snap)
        size_desc = "base model"

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[*] evaluating {len(rows)} samples")

    y_triaj_true, y_triaj_pred = [], []
    y_bolum_true, y_bolum_pred = [], []
    parse_ok_cnt = 0
    per_sample_times = []
    predictions = []

    for i, r in enumerate(rows):
        prompt = r["prompt"]
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768).to(DEVICE)
        t_s = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                temperature=1.0,
                top_p=1.0,
            )
        dt = time.time() - t_s
        per_sample_times.append(dt)

        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_output(gen)

        y_triaj_true.append(r["triaj"])
        y_triaj_pred.append(parsed["triaj"] or "UNKNOWN")
        y_bolum_true.append(r["bolum"])
        y_bolum_pred.append(parsed["bolum"] or "UNKNOWN")
        if parsed["parse_ok"]:
            parse_ok_cnt += 1

        predictions.append({
            "idx": i,
            "text": r["text"][:200],
            "true_triaj": r["triaj"],
            "true_bolum": r["bolum"],
            "pred_triaj": parsed["triaj"],
            "pred_bolum": parsed["bolum"],
            "pred_neden": parsed["neden"],
            "raw": gen[:400],
            "parse_ok": parsed["parse_ok"],
            "latency_sec": round(dt, 3),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} done (avg latency {sum(per_sample_times)/len(per_sample_times)*1000:.1f}ms)")

    if DEVICE == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
    else:
        import psutil
        peak_mem = psutil.Process().memory_info().rss / 1e6

    acc_triaj = accuracy_score(y_triaj_true, y_triaj_pred)
    f1_macro_triaj = f1_score(y_triaj_true, y_triaj_pred, average="macro",
                              labels=TRIAJ_CLASSES, zero_division=0)
    f1_weighted_triaj = f1_score(y_triaj_true, y_triaj_pred, average="weighted",
                                 labels=TRIAJ_CLASSES, zero_division=0)

    fgr_total = sum(1 for t, p in zip(y_triaj_true, y_triaj_pred)
                    if t in ("Kirmizi", "Sari") and p == "Yesil")
    fgr_base = sum(1 for t in y_triaj_true if t in ("Kirmizi", "Sari"))
    fgr = fgr_total / max(1, fgr_base)

    fgr_red_only = sum(1 for t, p in zip(y_triaj_true, y_triaj_pred)
                       if t == "Kirmizi" and p != "Kirmizi")
    fgr_red_denom = sum(1 for t in y_triaj_true if t == "Kirmizi")
    fnr_red = fgr_red_only / max(1, fgr_red_denom)

    acc_bolum = accuracy_score(y_bolum_true, y_bolum_pred)
    f1_macro_bolum = f1_score(y_bolum_true, y_bolum_pred, average="macro", zero_division=0)
    f1_weighted_bolum = f1_score(y_bolum_true, y_bolum_pred, average="weighted", zero_division=0)

    report_triaj = classification_report(
        y_triaj_true, y_triaj_pred, labels=TRIAJ_CLASSES, zero_division=0, digits=3
    )

    results = {
        "tag": args.tag,
        "model_id": args.model_id,
        "adapter": args.adapter,
        "size_mb": round(size_mb, 1),
        "size_desc": size_desc,
        "peak_gpu_mem_mb": round(peak_mem, 1),
        "load_time_sec": round(t_load, 2),
        "total_samples": len(rows),
        "parse_ok_count": parse_ok_cnt,
        "parse_ok_rate": round(parse_ok_cnt / max(1, len(rows)), 4),
        "avg_latency_ms": round(sum(per_sample_times) / len(per_sample_times) * 1000, 2),
        "p50_latency_ms": round(sorted(per_sample_times)[len(per_sample_times) // 2] * 1000, 2),
        "p95_latency_ms": round(sorted(per_sample_times)[int(len(per_sample_times) * 0.95)] * 1000, 2),
        "triaj": {
            "accuracy": round(acc_triaj, 4),
            "f1_macro": round(f1_macro_triaj, 4),
            "f1_weighted": round(f1_weighted_triaj, 4),
            "false_green_rate": round(fgr, 4),
            "false_negative_rate_red": round(fnr_red, 4),
        },
        "bolum": {
            "accuracy": round(acc_bolum, 4),
            "f1_macro": round(f1_macro_bolum, 4),
            "f1_weighted": round(f1_weighted_bolum, 4),
        },
        "classification_report_triaj": report_triaj,
    }

    with (out_dir / f"{args.tag}_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (out_dir / f"{args.tag}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n=== Results [{args.tag}] ===")
    print(f"  parse_ok: {results['parse_ok_rate']*100:.1f}%")
    print(f"  latency: {results['avg_latency_ms']:.1f}ms (p95={results['p95_latency_ms']:.1f}ms)")
    print(f"  size: {results['size_mb']}MB ({size_desc})")
    print(f"  peak GPU: {results['peak_gpu_mem_mb']:.0f}MB")
    print(f"  TRIAJ acc={results['triaj']['accuracy']} macro-F1={results['triaj']['f1_macro']} "
          f"weighted-F1={results['triaj']['f1_weighted']} FGR={results['triaj']['false_green_rate']} "
          f"FNR-Red={results['triaj']['false_negative_rate_red']}")
    print(f"  BOLUM acc={results['bolum']['accuracy']} macro-F1={results['bolum']['f1_macro']} "
          f"weighted-F1={results['bolum']['f1_weighted']}")


if __name__ == "__main__":
    main()
