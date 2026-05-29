#!/usr/bin/env python3
"""
cascade_eval.py — Run 3-layer cascade on test set and compute metrics.

Output identical schema to evaluate.py (results json + predictions jsonl).

Usage:
  python eval/cascade_eval.py \
      --l1_adapter models/smollm2_360m_qlora/final_adapter \
      --l2_adapter models/qwen25_15b_lora_balanced/final_adapter \
      --l3_adapter models/gemma4_e2b_qlora/final_adapter \
      --tag cascade_balanced
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


INSTRUCTION_L12 = (
    "Asagidaki hasta semptom metnini analiz et. Triaj seviyesini "
    "(Yesil/Sari/Kirmizi) ve en uygun tibbi bolumu belirle. "
    "Cevabini SADECE gecerli bir JSON objesi olarak ver."
)

def fp(text: str) -> str:
    return f"### Gorev\n{INSTRUCTION_L12}\n\n### Semptom\n{text.strip()}\n\n### Cevap\n"


def fp3(text: str, prev: dict) -> str:
    """L3 prompt: matches training format (### Gorev / ### Semptom / ### Cevap).
    Prev info injected as part of Gorev instruction so Gemma sees in-distribution structure.
    """
    prev_str = json.dumps(prev, ensure_ascii=False)
    instruction = (
        f"{INSTRUCTION_L12} Onceki on-analiz cikti: {prev_str}. "
        "Bu cikti yanlis olabilir, semptoma gore duzelt veya onayla."
    )
    return f"### Gorev\n{instruction}\n\n### Semptom\n{text.strip()}\n\n### Cevap\n"


def parse_json(txt: str) -> dict:
    m = re.search(r"\{[^{}]*\}", txt, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        triaj = (obj.get("triaj") or "").strip()
        if triaj.lower() in ["yeşil", "yesil", "green"]: triaj = "Yesil"
        elif triaj.lower() in ["sarı", "sari", "yellow"]: triaj = "Sari"
        elif triaj.lower() in ["kırmızı", "kirmizi", "red"]: triaj = "Kirmizi"
        obj["triaj"] = triaj
        return obj
    except Exception:
        return {}


def load(model_id: str, adapter):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter and Path(adapter).exists():
        m = PeftModel.from_pretrained(m, adapter)
        m = m.merge_and_unload()
    m.eval()
    return tok, m


def gen(tok, model, prompt: str, max_new=120) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, temperature=1.0, top_p=1.0,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1_id", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--l1_adapter", default="models/smollm2_360m_qlora/final_adapter")
    ap.add_argument("--l2_id", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--l2_adapter", default="models/qwen25_15b_lora_balanced/final_adapter")
    ap.add_argument("--l3_id", default="google/gemma-4-E2B")
    ap.add_argument("--l3_adapter", default="models/gemma4_e2b_qlora_balanced/final_adapter")
    ap.add_argument("--test", default="data/labeled/test.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] L1: {args.l1_id} + {args.l1_adapter}")
    l1_tok, l1 = load(args.l1_id, args.l1_adapter)
    print(f"[*] L2: {args.l2_id} + {args.l2_adapter}")
    l2_tok, l2 = load(args.l2_id, args.l2_adapter)
    print(f"[*] L3: {args.l3_id} + {args.l3_adapter}")
    l3_tok, l3 = load(args.l3_id, args.l3_adapter)

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[*] cascade eval on {len(rows)} samples")

    y_t_true, y_t_pred = [], []
    y_b_true, y_b_pred = [], []
    layer_counts = {"L1": 0, "L1+L2": 0, "L1+L2+L3": 0}
    latencies = []
    parse_ok_cnt = 0
    preds = []

    for i, r in enumerate(rows):
        text = r["text"]
        t0 = time.time()
        r1 = parse_json(gen(l1_tok, l1, fp(text), max_new=80))
        triaj_l1 = r1.get("triaj", "")

        if triaj_l1 == "Yesil":
            final_t, final_b = "Yesil", r1.get("bolum", "")
            layers = "L1"
        else:
            r2 = parse_json(gen(l2_tok, l2, fp(text), max_new=120))
            triaj_l2 = r2.get("triaj", "")
            bolum_l2 = r2.get("bolum", "")
            if triaj_l2 == "Yesil":
                final_t, final_b = "Yesil", bolum_l2
                layers = "L1+L2"
            else:
                prev = {"triaj": triaj_l2 or triaj_l1, "bolum": bolum_l2 or r1.get("bolum", "")}
                r3 = parse_json(gen(l3_tok, l3, fp3(text, prev), max_new=150))
                final_t = r3.get("triaj") or triaj_l2 or triaj_l1 or "Sari"
                final_b = r3.get("bolum") or bolum_l2 or ""
                layers = "L1+L2+L3"

        dt = time.time() - t0
        latencies.append(dt)
        layer_counts[layers] += 1

        if final_t in TRIAJ_CLASSES:
            parse_ok_cnt += 1

        y_t_true.append(r["triaj"])
        y_t_pred.append(final_t if final_t in TRIAJ_CLASSES else "UNKNOWN")
        y_b_true.append(r["bolum"])
        y_b_pred.append(final_b or "UNKNOWN")

        preds.append({
            "idx": i,
            "text": r["text"][:200],
            "true_triaj": r["triaj"],
            "true_bolum": r["bolum"],
            "pred_triaj": final_t,
            "pred_bolum": final_b,
            "layers": layers,
            "latency_sec": round(dt, 3),
        })

        if (i + 1) % 50 == 0:
            avg = sum(latencies) / len(latencies) * 1000
            print(f"  {i+1}/{len(rows)} done  avg={avg:.0f}ms  layers={layer_counts}")

    acc_t = accuracy_score(y_t_true, y_t_pred)
    f1m_t = f1_score(y_t_true, y_t_pred, average="macro", labels=TRIAJ_CLASSES, zero_division=0)
    f1w_t = f1_score(y_t_true, y_t_pred, average="weighted", labels=TRIAJ_CLASSES, zero_division=0)
    fgr = sum(1 for t, p in zip(y_t_true, y_t_pred)
              if t in ("Kirmizi", "Sari") and p == "Yesil") / max(1, sum(1 for t in y_t_true if t in ("Kirmizi", "Sari")))
    fnr_red = sum(1 for t, p in zip(y_t_true, y_t_pred)
                  if t == "Kirmizi" and p != "Kirmizi") / max(1, sum(1 for t in y_t_true if t == "Kirmizi"))
    acc_b = accuracy_score(y_b_true, y_b_pred)
    f1m_b = f1_score(y_b_true, y_b_pred, average="macro", zero_division=0)
    f1w_b = f1_score(y_b_true, y_b_pred, average="weighted", zero_division=0)
    report_t = classification_report(y_t_true, y_t_pred, labels=TRIAJ_CLASSES, zero_division=0, digits=3)

    results = {
        "tag": args.tag,
        "models": {"l1": args.l1_id, "l2": args.l2_id, "l3": args.l3_id},
        "adapters": {"l1": args.l1_adapter, "l2": args.l2_adapter, "l3": args.l3_adapter},
        "total_samples": len(rows),
        "parse_ok_count": parse_ok_cnt,
        "parse_ok_rate": round(parse_ok_cnt / max(1, len(rows)), 4),
        "layer_counts": layer_counts,
        "layer_pct": {k: round(v / len(rows) * 100, 1) for k, v in layer_counts.items()},
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2] * 1000, 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2),
        "triaj": {
            "accuracy": round(acc_t, 4),
            "f1_macro": round(f1m_t, 4),
            "f1_weighted": round(f1w_t, 4),
            "false_green_rate": round(fgr, 4),
            "false_negative_rate_red": round(fnr_red, 4),
        },
        "bolum": {
            "accuracy": round(acc_b, 4),
            "f1_macro": round(f1m_b, 4),
            "f1_weighted": round(f1w_b, 4),
        },
        "classification_report_triaj": report_t,
    }

    with (out_dir / f"{args.tag}_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (out_dir / f"{args.tag}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n=== Results [{args.tag}] ===")
    print(f"  parse_ok: {results['parse_ok_rate']*100:.1f}%")
    print(f"  layers: {results['layer_pct']}")
    print(f"  latency avg={results['avg_latency_ms']:.0f}ms p95={results['p95_latency_ms']:.0f}ms")
    print(f"  TRIAJ acc={results['triaj']['accuracy']} macro-F1={results['triaj']['f1_macro']} "
          f"FGR={results['triaj']['false_green_rate']} FNR-Red={results['triaj']['false_negative_rate_red']}")
    print(f"  BOLUM acc={results['bolum']['accuracy']} macro-F1={results['bolum']['f1_macro']}")


if __name__ == "__main__":
    main()
