#!/usr/bin/env python3
"""
sample_data.py — Random sampler for 2 HF Turkish medical datasets.

Samples 5k rows from each:
  - alibayram/doktorsitesi  (question_content, doctor_speciality)
  - alibayram/turkish-hospital-medical-articles

Outputs:
  data/sampled/doktorsitesi_5k.jsonl
  data/sampled/hospital_articles_5k.jsonl

Uses seed=42 for reproducibility. Normalizes speciality slugs to main branches.
Writes UTF-8 encoded JSONL.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

# Map doktorsitesi slug → main branch
BOLUM_MAP = [
    (r"kadin-hastaliklari|dogum|jinekoloji|gebelik|ureme-endokrin|infertilite|perinatoloji",
     "Kadin Dogum"),
    (r"cocuk-sagligi|cocuk-noroloji|cocuk-kardiyoloji|cocuk-gogus|cocuk-urolojisi|"
     r"cocuk-cerrahi|cocuk-hematoloji|cocuk-endokrin|cocuk-gastroenteroloji|cocuk-immunoloji|"
     r"neonatoloji|pediatri",
     "Cocuk Sagligi"),
    (r"kardiyoloji|kardiyovaskuler|kalp",
     "Kardiyoloji"),
    (r"noroloji|beyin-ve-sinir-cerrahisi|norofizyoloji|norosirurji",
     "Noroloji"),
    (r"ortopedi|travmatoloji|el-cerrahisi|mikrocerrahi",
     "Ortopedi"),
    (r"uroloji|androloji",
     "Uroloji"),
    (r"dahiliye|ic-hastaliklari|gastroenteroloji|nefroloji|endokrinoloji|metabolizma|"
     r"hematoloji|romatoloji|geriatri|tibbi-onkoloji",
     "Dahiliye"),
    (r"genel-cerrahi|cerrahi-onkoloji",
     "Genel Cerrahi"),
    (r"gogus-cerrahi|gogus-hastaliklari",
     "Gogus Hastaliklari"),
    (r"kulak-burun-bogaz|kbb",
     "KBB"),
    (r"goz-hastaliklari|oftalmoloji",
     "Goz Hastaliklari"),
    (r"dermatoloji|cildiye",
     "Dermatoloji"),
    (r"psikiyatri",
     "Psikiyatri"),
    (r"psikoloji",
     "Psikoloji"),
    (r"plastik|rekonstruktif|estetik",
     "Plastik Cerrahi"),
    (r"fiziksel-tip|rehabilitasyon|fizik-tedavi",
     "Fizik Tedavi"),
    (r"enfeksiyon|infeksiyon",
     "Enfeksiyon"),
    (r"radyoloji",
     "Radyoloji"),
    (r"anestezi|algoloji",
     "Anestezi"),
    (r"dis|agiz|ortodonti|periodontoloji|endodonti",
     "Dis Hekimligi"),
    (r"beslenme|diyetetik|diyetisyen",
     "Beslenme"),
    (r"aile-hekim",
     "Aile Hekimligi"),
    (r"acil",
     "Acil Tip"),
]


def normalize_bolum(spec_slug: str) -> str:
    if not spec_slug:
        return "Diger"
    s = spec_slug.lower()
    for pattern, name in BOLUM_MAP:
        if re.search(pattern, s):
            return name
    return "Diger"


def sample_doktorsitesi(out_path: Path, n: int, seed: int) -> dict:
    ds = load_dataset("alibayram/doktorsitesi")
    split = ds["train"]
    idx = list(range(len(split)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    kept = []
    for i in idx:
        row = split[i]
        text = (row.get("question_content") or "").strip()
        spec = (row.get("doctor_speciality") or "").strip()
        if not text or len(text) < 20:
            continue
        bolum = normalize_bolum(spec)
        if bolum == "Diger":
            continue
        kept.append({
            "source": "doktorsitesi",
            "text": text,
            "bolum_raw": spec,
            "bolum": bolum,
            "answer": (row.get("question_answer") or "").strip(),
        })
        if len(kept) >= n:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(r["bolum"] for r in kept)
    return {"total": len(kept), "bolum_dist": dict(dist.most_common())}


def sample_hospital(out_path: Path, n: int, seed: int) -> dict:
    ds = load_dataset("alibayram/turkish-hospital-medical-articles")
    split = ds[list(ds.keys())[0]]
    idx = list(range(len(split)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    cols = split.column_names
    text_col = next((c for c in ["content", "article", "text", "body"] if c in cols), cols[0])
    spec_col = next((c for c in ["speciality", "category", "department", "bolum"] if c in cols), None)

    kept = []
    for i in idx:
        row = split[i]
        text = (row.get(text_col) or "").strip()
        if not text or len(text) < 50:
            continue
        spec = (row.get(spec_col) or "").strip() if spec_col else ""
        bolum = normalize_bolum(spec) if spec else "Diger"
        kept.append({
            "source": "hospital_articles",
            "text": text[:2000],
            "bolum_raw": spec,
            "bolum": bolum,
            "answer": "",
        })
        if len(kept) >= n:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(r["bolum"] for r in kept)
    return {"total": len(kept), "bolum_dist": dict(dist.most_common()), "text_col": text_col, "spec_col": spec_col}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/sampled")
    ap.add_argument("--dataset", choices=["doktorsitesi", "hospital", "both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if args.dataset in ("doktorsitesi", "both"):
        print(f"[*] Sampling {args.n} from doktorsitesi (seed={args.seed})...")
        summary["doktorsitesi"] = sample_doktorsitesi(
            out_dir / "doktorsitesi_5k.jsonl", args.n, args.seed
        )
        print(f"    saved: {out_dir / 'doktorsitesi_5k.jsonl'}")
        print(f"    total: {summary['doktorsitesi']['total']}")
        print(f"    bolum dist: {summary['doktorsitesi']['bolum_dist']}")

    if args.dataset in ("hospital", "both"):
        print(f"\n[*] Sampling {args.n} from hospital articles (seed={args.seed})...")
        try:
            summary["hospital"] = sample_hospital(
                out_dir / "hospital_articles_5k.jsonl", args.n, args.seed
            )
            print(f"    saved: {out_dir / 'hospital_articles_5k.jsonl'}")
            print(f"    total: {summary['hospital']['total']}")
            print(f"    text_col: {summary['hospital']['text_col']} | spec_col: {summary['hospital']['spec_col']}")
            print(f"    bolum dist: {summary['hospital']['bolum_dist']}")
        except Exception as e:
            print(f"    ERROR (hospital): {e}", file=sys.stderr)

    with (out_dir / "sample_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
