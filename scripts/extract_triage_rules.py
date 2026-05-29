#!/usr/bin/env python3
"""
extract_triage_rules.py — Extract symptom→triage keyword rules from Kaggle data.

Maps Kaggle urgency_label to 3 classes:
  NORMAL           → Yesil
  ACIL / ACIL      → Sari
  COK ACIL         → Kirmizi

Builds keyword frequency tables per class and exports:
  configs/triage_rules.json (keyword lists + scores)
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def turkish_normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("i̇", "i").replace("İ", "i")
    replacements = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


LABEL_TO_TRIAGE = {
    "NORMAL": "Yesil",
    "ACIL": "Sari",
    "ACİL": "Sari",
    "COK ACIL": "Kirmizi",
    "ÇOK ACİL": "Kirmizi",
}


def main():
    src = Path("data/raw/kaggle_triage/medical_data.json")
    out = Path("configs/triage_rules.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    symptom_counts = defaultdict(Counter)
    text_counts = defaultdict(Counter)
    label_totals = Counter()
    examples = defaultdict(list)

    for row in data:
        triage = LABEL_TO_TRIAGE.get(row["urgency_label"])
        if not triage:
            continue
        label_totals[triage] += 1
        for s in row.get("symptoms", []):
            symptom_counts[triage][turkish_normalize(s)] += 1
        tokens = re.findall(r"[a-zçğıöşü]+", row.get("input_text", "").lower())
        for t in tokens:
            if len(t) >= 4:
                text_counts[triage][turkish_normalize(t)] += 1
        if len(examples[triage]) < 6:
            examples[triage].append({
                "text": row["input_text"][:200],
                "symptoms": row["symptoms"][:6],
                "reasoning": row.get("reasoning", "")[:200],
            })

    def discriminative(target: str, others: list, counter_map: dict, min_count=3) -> list:
        total_target = label_totals[target]
        target_c = counter_map[target]
        out = []
        for tok, c in target_c.items():
            if c < min_count:
                continue
            p_target = c / total_target
            p_others = sum(counter_map[o].get(tok, 0) for o in others) / max(1, sum(label_totals[o] for o in others))
            score = p_target / (p_others + 0.001)
            out.append((tok, c, round(score, 2)))
        out.sort(key=lambda x: (-x[2], -x[1]))
        return out[:80]

    kirmizi_sym = discriminative("Kirmizi", ["Sari", "Yesil"], symptom_counts)
    sari_sym = discriminative("Sari", ["Kirmizi", "Yesil"], symptom_counts)
    yesil_sym = discriminative("Yesil", ["Kirmizi", "Sari"], symptom_counts)

    kirmizi_txt = discriminative("Kirmizi", ["Sari", "Yesil"], text_counts, min_count=5)
    sari_txt = discriminative("Sari", ["Kirmizi", "Yesil"], text_counts, min_count=5)
    yesil_txt = discriminative("Yesil", ["Kirmizi", "Sari"], text_counts, min_count=5)

    critical_phrases = [
        "bilinc kayb", "bilinc bulanik", "nefes alam", "nefes daral", "gogus agris",
        "gogus sikism", "kol uyus", "yuz felc", "konuş bozuk", "konusam",
        "kan geliyor", "kanama", "kusarak kan", "morar", "siyanoz",
        "baygin", "bayilma", "konvulz", "havale", "nobet",
        "sert boyun", "ensede sert", "ani bas agris", "siddetli bas agris",
        "siddetli karin", "karin akut", "karın akut",
        "kalp kriz", "felc", "inme", "inmem", "stroke",
        "112", "acil servis",
    ]
    mild_phrases = [
        "hafif", "2 gun", "3 gun", "gecti", "arada bir", "ara ara",
        "muayene", "kontrole", "rutin", "randevu",
        "kroniğim", "kronik", "yavas yavas",
    ]

    triage_summary = {
        "label_totals": dict(label_totals),
        "mapping": LABEL_TO_TRIAGE,
        "critical_phrases": critical_phrases,
        "mild_phrases": mild_phrases,
        "symptom_keywords": {
            "Kirmizi": kirmizi_sym,
            "Sari": sari_sym,
            "Yesil": yesil_sym,
        },
        "text_keywords": {
            "Kirmizi": kirmizi_txt,
            "Sari": sari_txt,
            "Yesil": yesil_txt,
        },
        "examples": {k: v for k, v in examples.items()},
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(triage_summary, f, ensure_ascii=False, indent=2)

    print(f"[+] saved {out}")
    print("label totals:", dict(label_totals))
    print("\nTop Kirmizi symptoms:", [s[0] for s in kirmizi_sym[:15]])
    print("Top Sari symptoms:", [s[0] for s in sari_sym[:15]])
    print("Top Yesil symptoms:", [s[0] for s in yesil_sym[:15]])


if __name__ == "__main__":
    main()
