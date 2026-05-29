#!/usr/bin/env python3
"""
generate_report.py — Aggregate eval results into REPORT.md comparison table.

Reads eval/results/*_results.json and writes REPORT.md aligned with
hoca project-plan structure (Section 4 / 7 / 8).
"""
import json
from pathlib import Path
from datetime import date


TAGS_ORDER = [
    ("smollm2_zeroshot", "SmolLM2-360M (zero-shot)"),
    ("smollm2_ft", "SmolLM2-360M (QLoRA)"),
    ("qwen25_zeroshot", "Qwen2.5-1.5B (zero-shot)"),
    ("qwen25_ft", "Qwen2.5-1.5B (QLoRA)"),
    ("qwen25_lora_ft", "Qwen2.5-1.5B (LoRA bf16)"),
    ("gemma4_zeroshot", "Gemma-4-E2B (zero-shot)"),
    ("gemma4_ft", "Gemma-4-E2B (QLoRA)"),
]


def load_results(results_dir: Path) -> dict:
    out = {}
    for f in results_dir.glob("*_results.json"):
        tag = f.stem.replace("_results", "")
        out[tag] = json.loads(f.read_text(encoding="utf-8"))
    return out


def fmt(val, suffix=""):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.3f}{suffix}"
    return f"{val}{suffix}"


def section_header():
    return f"""# Tibbi Triaj ve Bolum Yonlendirme — Deneysel Rapor

**Tarih:** {date.today().isoformat()}
**Donanim:** NVIDIA DGX GB10 (Blackwell, bf16/fp8)
**Kullanici Bilgilendirme:** *Bu sistem sadece bir on yonlendirme aracidir, tibbi tani koymaz. Acil durumlarda 112'yi arayin.*

---

## 1. Ozet

Bu raporda 3 farkli dil modelinin Turkce tibbi triaj ve bolum yonlendirme gorevinde performansi karsilastirilmaktadir. Modeller QLoRA ile fine-tune edilmis olup, zero-shot baseline ve fine-tuned sonuclari raporlanmistir. Karsilastirma amacli Qwen2.5-1.5B uzerinde LoRA (bf16, quantizasyonsuz) subset deneyi de dahil edilmistir.

**Modeller:**
- Layer 1 (hizli filtre): HuggingFaceTB/SmolLM2-360M
- Layer 2 (ana siniflandirici): Qwen/Qwen2.5-1.5B
- Layer 3 (reasoning): google/gemma-4-E2B

**Veri:**
- alibayram/doktorsitesi (HF) — 5000 ornek, semptom→bolum
- alibayram/turkish-hospital-medical-articles (HF) — (gated, erisime gore)
- Kaggle `doguseryarar/turkish-medical-emergency-triage-dataset` — triaj kural kaynagi (920 ornek)

"""


def results_table(results: dict) -> str:
    header = "| Model | Boyut (MB) | Peak GPU (MB) | Parse OK % | Inf Latency (ms) | Triaj Acc | Triaj Macro-F1 | Triaj Weighted-F1 | **FGR** | FNR-Red | Bolum Acc | Bolum Macro-F1 |\n"
    header += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for tag, label in TAGS_ORDER:
        r = results.get(tag)
        if not r:
            continue
        rows.append(
            f"| {label} "
            f"| {fmt(r.get('size_mb'))} "
            f"| {fmt(r.get('peak_gpu_mem_mb'))} "
            f"| {fmt(r.get('parse_ok_rate', 0) * 100, '%')} "
            f"| {fmt(r.get('avg_latency_ms'))} "
            f"| {fmt(r['triaj']['accuracy'])} "
            f"| {fmt(r['triaj']['f1_macro'])} "
            f"| {fmt(r['triaj']['f1_weighted'])} "
            f"| **{fmt(r['triaj']['false_green_rate'])}** "
            f"| {fmt(r['triaj']['false_negative_rate_red'])} "
            f"| {fmt(r['bolum']['accuracy'])} "
            f"| {fmt(r['bolum']['f1_macro'])} |"
        )
    return header + "\n".join(rows)


def main():
    results_dir = Path("eval/results")
    results = load_results(results_dir)
    if not results:
        print("[!] no results found")
        return

    out = section_header()
    out += "## 2. Karsilastirmali Tablo\n\n"
    out += results_table(results) + "\n\n"

    out += "## 3. Metrik Tanimlari\n\n"
    out += (
        "- **Boyut:** Base model/adapter'in disk uzerindeki buyuklugu (MB).\n"
        "- **Peak GPU:** Inference sirasindaki maksimum VRAM kullanimi (MB).\n"
        "- **Parse OK %:** Modelin gecerli JSON ciktisi uretebildigi orneklerin orani.\n"
        "- **Inf Latency:** Ornek basina ortalama inference suresi (ms).\n"
        "- **Triaj/Bolum Acc / Macro-F1 / Weighted-F1:** Standart siniflandirma metrikleri.\n"
        "- **FGR (False Green Rate):** Gercekte Kirmizi veya Sari olan vakalarin Yesil olarak "
        "siniflandirilma orani — tibbi guvenlik icin kritik metrik, dusuk olmali.\n"
        "- **FNR-Red:** Gercekte Kirmizi olan vakalarin Kirmizi olmayan siniflara atama orani.\n\n"
    )

    out += "## 4. Veri Bolunmesi\n\n"
    out += (
        "80/10/10 stratified split (triaj+bolum birlikte stratify), seed=42.\n\n"
        "- Train: 3972\n- Val: 475\n- Test: 553\n\n"
    )

    out += "## 5. Cascade Inference\n\n"
    out += (
        "3 katmanli kaskad: L1 (SmolLM2) → L2 (Qwen) → L3 (Gemma-4).\n"
        "- L1 Yesil cevap verirse burada dur (hiz odakli).\n"
        "- L1 Yesil disi cevaplarsa L2 full siniflandirma yapar.\n"
        "- L2 Yesil disi kararlarda L3 reasoning (neden) ve dogrulama gecer.\n"
        "- Tum ciktilar zorunlu disclaimer ile sunulur.\n\n"
    )

    out += "## 6. Lisans ve Guvenlik\n\n"
    out += (
        "- Doktorsitesi: Apache 2.0\n"
        "- Turkish-Hospital-Medical-Articles: CC BY 4.0\n"
        "- Kaggle emergency triage: bilgi amacli, sadece kural cikarimi\n"
        "- Sistem tibbi tani koymaz, sadece on yonlendirme amaclidir\n"
    )

    Path("REPORT.md").write_text(out, encoding="utf-8")
    print(f"[+] REPORT.md yazildi ({len(out)} karakter)")


if __name__ == "__main__":
    main()
