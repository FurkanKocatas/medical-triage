#!/usr/bin/env python3
"""
split_and_format.py — 80/10/10 split + SFT instruction format.

Inputs:
  data/labeled/labeled_all.jsonl

Outputs:
  data/labeled/train.jsonl
  data/labeled/val.jsonl
  data/labeled/test.jsonl
  data/labeled/split_stats.json

Each record: {"prompt": ..., "response": ..., "triaj": ..., "bolum": ..., "text": ...}
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

INSTRUCTION = (
    "Asagidaki hasta semptom metnini analiz et. Triaj seviyesini "
    "(Yesil/Sari/Kirmizi) ve en uygun tibbi bolumu belirle. "
    "Cevabini SADECE gecerli bir JSON objesi olarak ver."
)

DISCLAIMER = (
    "Bu sistem sadece bir on yonlendirme aracidir, tibbi tani koymaz. "
    "Acil durumlarda 112'yi arayin."
)


def format_prompt(text: str) -> str:
    return (
        f"### Gorev\n{INSTRUCTION}\n\n"
        f"### Semptom\n{text.strip()}\n\n"
        f"### Cevap\n"
    )


def format_response(triaj: str, bolum: str, neden: str) -> str:
    obj = {"triaj": triaj, "bolum": bolum, "neden": neden}
    return json.dumps(obj, ensure_ascii=False)


def stratified_split(rows: list, ratios=(0.8, 0.1, 0.1), seed=42) -> tuple:
    random.Random(seed).shuffle(rows)
    groups = {}
    for r in rows:
        key = (r["triaj"], r["bolum"])
        groups.setdefault(key, []).append(r)

    train, val, test = [], [], []
    for key, grp in groups.items():
        n = len(grp)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train += grp[:n_train]
        val += grp[n_train:n_train + n_val]
        test += grp[n_train + n_val:]
    rng = random.Random(seed)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/labeled/labeled_all.jsonl")
    ap.add_argument("--out-dir", default="data/labeled")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-text-chars", type=int, default=1200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).open("r", encoding="utf-8")]
    print(f"[*] loaded {len(rows)} labeled rows")

    formatted = []
    for r in rows:
        text = r["text"][:args.max_text_chars]
        formatted.append({
            "source": r["source"],
            "triaj": r["triaj"],
            "bolum": r["bolum"],
            "text": text,
            "prompt": format_prompt(text),
            "response": format_response(r["triaj"], r["bolum"], r["neden"]),
            "full": format_prompt(text) + format_response(r["triaj"], r["bolum"], r["neden"]),
        })

    train, val, test = stratified_split(formatted, seed=args.seed)

    out = Path(args.out_dir)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[+] {p}: {len(split)}")

    def dist(split):
        return {
            "total": len(split),
            "triaj": dict(Counter(r["triaj"] for r in split)),
            "source": dict(Counter(r["source"] for r in split)),
            "bolum_top10": dict(Counter(r["bolum"] for r in split).most_common(10)),
        }

    stats = {
        "train": dist(train),
        "val": dist(val),
        "test": dist(test),
        "instruction": INSTRUCTION,
        "disclaimer": DISCLAIMER,
        "max_text_chars": args.max_text_chars,
        "seed": args.seed,
    }
    with (out / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[+] stats saved to {out / 'split_stats.json'}")
    print(f"    train triaj: {stats['train']['triaj']}")
    print(f"    val   triaj: {stats['val']['triaj']}")
    print(f"    test  triaj: {stats['test']['triaj']}")


if __name__ == "__main__":
    main()
