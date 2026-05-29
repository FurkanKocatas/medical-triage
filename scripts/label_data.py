#!/usr/bin/env python3
"""
label_data.py — Apply Kaggle-derived triage rules to sampled HF data.

Inputs:
  data/sampled/doktorsitesi_5k.jsonl
  data/sampled/hospital_articles_5k.jsonl (if present)
  configs/triage_rules.json

Output:
  data/labeled/labeled_all.jsonl

Each row gets: triaj (Yesil/Sari/Kirmizi), bolum, neden, + originals.

Rule priority:
  1. Critical phrase match anywhere  → Kirmizi (force)
  2. Weighted keyword scoring        → argmax class
  3. Mild phrase present + low score → Yesil bias
  4. Tie/zero score                  → Yesil default for info/article text,
                                        Sari default for symptom text
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def turkish_normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("İ", "i").replace("i̇", "i")
    for k, v in {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def load_rules(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_scorers(rules: dict) -> dict:
    def to_weighted(lst):
        return {tok: min(score, 15.0) for tok, _cnt, score in lst}

    return {
        "Kirmizi": {
            **to_weighted(rules["symptom_keywords"]["Kirmizi"]),
            **{k: v * 0.6 for k, v in to_weighted(rules["text_keywords"]["Kirmizi"]).items()},
        },
        "Sari": {
            **to_weighted(rules["symptom_keywords"]["Sari"]),
            **{k: v * 0.6 for k, v in to_weighted(rules["text_keywords"]["Sari"]).items()},
        },
        "Yesil": {
            **to_weighted(rules["symptom_keywords"]["Yesil"]),
            **{k: v * 0.6 for k, v in to_weighted(rules["text_keywords"]["Yesil"]).items()},
        },
    }


def score_text(norm_text: str, scorers: dict) -> dict:
    scores = {"Kirmizi": 0.0, "Sari": 0.0, "Yesil": 0.0}
    for cls, toks in scorers.items():
        for tok, w in toks.items():
            if tok in norm_text:
                scores[cls] += w
    return scores


ACUTE_MARKERS = [
    r"\bani\b", r"\baniden\b", r"\bbirdenbire\b", r"\bsu an\b", r"simdi basladi",
    r"yeni basladi", r"bugun basladi", r"saat once baslad", r"dakika once baslad",
    r"cok siddetli", r"dayanilmaz", r"\bcekilmiyor\b", r"cok kotulesti",
    r"acil servis", r"\bambulans\b", r"\b112\b", r"hastaneye git",
]

STRONG_RED_PATTERNS = [
    r"bilinc kayb",
    r"bilinc bulanik",
    r"\bkalp krizi gecir(iy|ir|ec)",
    r"kalp krizi belirti",
    r"\binme gecir",
    r"ani gelisen felc",
    r"yuzumde felc",
    r"ani yuzde felc",
    r"yuzde uyus",
    r"tek kolda uyus",
    r"ani konusma bozuk",
    r"ani siddetli bas agri",
    r"ani siddetli karin agri",
    r"ani gorme kayb",
    r"ensede sert",
    r"\bsert ense\b",
    r"havale gecir",
    r"\bnobet gecir",
    r"\bkonvulziy",
    r"dudaklar morar",
    r"\bsiyanoz\b",
    r"septik sok",
    r"anafilaktik sok",
    r"\btansiyon dus(tu|uyor)",
    r"hemoptizi",
    r"ani kan gelm",
]

PAST_TENSE_MARKERS = [
    r"gecir(di|mis|di[mk]|ecektim)",
    r"olmustu",
    r"yil(lar)? once",
    r"sene(ler)? once",
    r"ay(lar)? once baslad",
    r"gecmiste",
    r"daha once",
]


def label_row(text: str, source: str, scorers: dict, critical_phrases: list,
              mild_phrases: list) -> tuple:
    norm = turkish_normalize(text)

    has_past = any(re.search(p, norm) for p in PAST_TENSE_MARKERS)

    strong_fired = [p for p in STRONG_RED_PATTERNS if re.search(p, norm)]
    if strong_fired and not has_past:
        return "Kirmizi", f"Cidiyet belirtisi: {strong_fired[0]}", strong_fired

    has_acute = any(re.search(m, norm) for m in ACUTE_MARKERS)
    fired_critical = [p for p in critical_phrases if p in norm]
    if fired_critical and has_acute and not has_past:
        return "Kirmizi", f"Akut baslangic + kritik belirti: {', '.join(fired_critical[:3])}", fired_critical

    scores = score_text(norm, scorers)
    total = sum(scores.values())

    if source == "hospital_articles":
        if any(p in norm for p in STRONG_RED_PHRASES):
            return "Kirmizi", "Bilgilendirme metninde acil belirti tanimi", []
        if scores["Sari"] > scores["Yesil"] * 2:
            return "Sari", "Bilgilendirme metni orta risk konusu", []
        return "Yesil", "Bilgilendirme/makale metni", []

    if total < 1.5:
        has_mild = any(p in norm for p in mild_phrases)
        if has_mild:
            return "Yesil", "Hafif/kronik belirti ifadeleri", []
        return "Sari", "Net acil belirti yok, genel tibbi soru — orta oncelik", []

    has_mild = any(p in norm for p in mild_phrases)
    kirmizi_score = scores["Kirmizi"]
    sari_score = scores["Sari"]
    yesil_score = scores["Yesil"]

    if has_mild and kirmizi_score < 10 and not has_acute:
        return "Yesil", "Hafif/kronik ifade, acil bulgu yok", []

    if kirmizi_score >= 15 and has_acute:
        return "Kirmizi", f"Yuksek kirmizi skor + akut baslangic (skor={kirmizi_score:.1f})", []

    if kirmizi_score >= 20 and kirmizi_score > sari_score * 1.5:
        return "Kirmizi", f"Baskin kirmizi skor (skor={kirmizi_score:.1f})", []

    if yesil_score >= sari_score and yesil_score >= kirmizi_score:
        return "Yesil", f"Dusuk aciliyet belirtileri baskin (skor={yesil_score:.1f}/{total:.1f})", []

    return "Sari", f"Orta duzey belirtiler, hekim degerlendirmesi onerilir (skor={sari_score:.1f}/{total:.1f})", []


def process_file(in_path: Path, scorers: dict, critical: list, mild: list,
                 out_rows: list) -> Counter:
    dist = Counter()
    if not in_path.exists():
        print(f"[!] skip missing: {in_path}")
        return dist
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            triaj, neden, fired = label_row(text, r["source"], scorers, critical, mild)
            out = {
                "source": r["source"],
                "text": text,
                "bolum": r["bolum"],
                "bolum_raw": r.get("bolum_raw", ""),
                "triaj": triaj,
                "neden": neden,
                "fired_critical": fired,
            }
            out_rows.append(out)
            dist[triaj] += 1
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampled-dir", default="data/sampled")
    ap.add_argument("--rules", default="configs/triage_rules.json")
    ap.add_argument("--out", default="data/labeled/labeled_all.jsonl")
    args = ap.parse_args()

    rules = load_rules(Path(args.rules))
    scorers = build_scorers(rules)
    critical = rules["critical_phrases"]
    mild = rules["mild_phrases"]

    sampled = Path(args.sampled_dir)
    out_rows = []

    print("[*] Labeling doktorsitesi...")
    d1 = process_file(sampled / "doktorsitesi_5k.jsonl", scorers, critical, mild, out_rows)
    print(f"    dist: {dict(d1)}")

    print("[*] Labeling hospital_articles (if present)...")
    d2 = process_file(sampled / "hospital_articles_5k.jsonl", scorers, critical, mild, out_rows)
    print(f"    dist: {dict(d2)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = Counter(r["triaj"] for r in out_rows)
    bolum = Counter(r["bolum"] for r in out_rows)
    print(f"\n[+] saved {out_path} ({len(out_rows)} rows)")
    print(f"    triaj: {dict(total)}")
    print(f"    bolum top10: {dict(bolum.most_common(10))}")

    stats = {
        "total_rows": len(out_rows),
        "triaj_dist": dict(total),
        "bolum_dist": dict(bolum.most_common()),
        "sources": dict(Counter(r["source"] for r in out_rows)),
    }
    with (out_path.parent / "label_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
