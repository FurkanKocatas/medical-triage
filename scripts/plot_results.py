#!/usr/bin/env python3
"""
plot_results.py — Generate confusion matrices + scatter + class dist plots.

Outputs PNG files to eval/plots/.

Usage:
  python scripts/plot_results.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


TRIAJ_CLASSES = ["Yesil", "Sari", "Kirmizi"]
TRIAJ_DISPLAY = ["Yesil\n(green)", "Sari\n(yellow)", "Kirmizi\n(red)"]
COLORS = {"Yesil": "#2ecc71", "Sari": "#f1c40f", "Kirmizi": "#e74c3c"}

RESULTS_DIR = Path("eval/results")
PLOTS_DIR = Path("eval/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Display names for paper
MODELS = {
    "smollm2_zeroshot": "SmolLM2-360M\n(zero-shot)",
    "qwen25_zeroshot": "Qwen2.5-1.5B\n(zero-shot)",
    "gemma4_zeroshot": "Gemma-4-E2B\n(zero-shot)",
    "smollm2_ft": "SmolLM2-360M\n(QLoRA-FT)",
    "qwen25_ft": "Qwen2.5-1.5B\n(QLoRA-FT)",
    "qwen25_lora_ft": "Qwen2.5-1.5B\n(LoRA-bf16-FT)",
    "gemma4_ft": "Gemma-4-E2B\n(QLoRA-FT)",
    "qwen25_lora_balanced": "Qwen2.5-1.5B\n(LoRA-bf16-BALANCED)",
    "cascade_balanced": "Cascade v1\n(L1+L2bal+L3)",
    "cascade_balanced_v2": "Cascade v2\n(format-fix + Gemma-bal)",
}


def load_preds(tag):
    path = RESULTS_DIR / f"{tag}_predictions.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def load_results(tag):
    path = RESULTS_DIR / f"{tag}_results.json"
    if not path.exists():
        return None
    return json.load(path.open(encoding="utf-8"))


def plot_confusion_matrix(tag, ax=None):
    preds = load_preds(tag)
    if preds is None:
        return None
    y_true = [p["true_triaj"] for p in preds]
    y_pred = [p.get("pred_triaj") if p.get("pred_triaj") in TRIAJ_CLASSES else "UNKNOWN" for p in preds]
    labels = TRIAJ_CLASSES + (["UNKNOWN"] if "UNKNOWN" in y_pred else [])
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(TRIAJ_CLASSES)))
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_yticklabels(TRIAJ_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(MODELS.get(tag, tag), fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        if i >= len(TRIAJ_CLASSES):
            continue
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold")
    return cm


def plot_all_confusion_matrices():
    tags = list(MODELS.keys())
    n = len(tags)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    for i, tag in enumerate(tags):
        plot_confusion_matrix(tag, ax=axes[i])
    for j in range(len(tags), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Triaj Confusion Matrices (553 test samples)", fontsize=13, y=1.001)
    plt.tight_layout()
    out = PLOTS_DIR / "confusion_matrices_all.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[+] saved {out}")


def plot_pareto_scatter():
    """Latency vs F1-macro scatter — Pareto frontier."""
    points = []
    for tag in MODELS:
        r = load_results(tag)
        if r is None:
            continue
        f1m = r.get("triaj", {}).get("f1_macro", 0)
        lat = r.get("avg_latency_ms", 0)
        fnr = r.get("triaj", {}).get("false_negative_rate_red", 1)
        points.append((tag, lat, f1m, fnr))

    fig, ax = plt.subplots(figsize=(9, 6))
    for tag, lat, f1m, fnr in points:
        is_balanced = "balanced" in tag or "cascade_balanced" in tag
        is_zs = "zeroshot" in tag
        marker = "o"
        size = 200 if is_balanced else 100
        color = "#27ae60" if is_balanced else ("#95a5a6" if is_zs else "#3498db")
        ax.scatter(lat, f1m, s=size, c=color, marker=marker,
                   edgecolors="black", linewidths=1.2, alpha=0.85, zorder=3)
        ax.annotate(MODELS[tag].replace("\n", " "), (lat, f1m),
                    xytext=(7, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Avg Latency (ms, log scale)", fontsize=11)
    ax.set_ylabel("F1-macro (Triaj)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title("Latency vs Quality Pareto Frontier", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.333, color="gray", linestyle=":", alpha=0.5, label="random baseline (3-class)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = PLOTS_DIR / "pareto_scatter.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[+] saved {out}")


def plot_class_dist():
    """Train/val/test class dist before/after balanced."""
    splits = {
        "train (orig)": {"Yesil": 1211, "Sari": 2599, "Kirmizi": 162},
        "train_balanced": {"Yesil": 1211, "Sari": 2599, "Kirmizi": 648},
        "val": {"Yesil": 142, "Sari": 318, "Kirmizi": 15},
        "test": {"Yesil": 174, "Sari": 345, "Kirmizi": 34},
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(splits))
    width = 0.27
    for i, c in enumerate(TRIAJ_CLASSES):
        vals = [splits[s][c] for s in splits]
        ax.bar(x + (i - 1) * width, vals, width, label=c, color=COLORS[c],
               edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(splits.keys(), fontsize=10)
    ax.set_ylabel("Sample count", fontsize=11)
    ax.set_title("Class Distribution: Original vs Balanced (oversample K x4)", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Add percent labels above bars
    for i, c in enumerate(TRIAJ_CLASSES):
        for j, s in enumerate(splits):
            tot = sum(splits[s].values())
            pct = splits[s][c] / tot * 100
            v = splits[s][c]
            ax.text(j + (i - 1) * width, v + 30, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / "class_distribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[+] saved {out}")


def plot_metric_comparison():
    """Bar chart: F1-macro + FNR-Red across models."""
    tags = ["smollm2_ft", "qwen25_ft", "qwen25_lora_ft", "gemma4_ft",
            "qwen25_lora_balanced", "cascade_balanced", "cascade_balanced_v2"]
    f1m_vals, fnr_vals, names = [], [], []
    for tag in tags:
        r = load_results(tag)
        if r is None:
            continue
        f1m_vals.append(r["triaj"]["f1_macro"])
        fnr_vals.append(r["triaj"]["false_negative_rate_red"])
        names.append(MODELS.get(tag, tag).replace("\n", " "))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    bars1 = ax1.bar(x, f1m_vals, color=["#3498db"] * 4 + ["#27ae60"] * (len(names) - 4),
                    edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("F1-macro", fontsize=11)
    ax1.set_title("F1-macro (Triaj) — Higher is better", fontsize=12)
    ax1.set_ylim(0, max(f1m_vals) * 1.2)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars1, f1m_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, fnr_vals, color=["#e74c3c"] * 4 + ["#f39c12"] * (len(names) - 4),
                    edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("FNR-Red (False Negative Rate, Kirmizi)", fontsize=11)
    ax2.set_title("FNR-Red — Lower is better (CRITICAL safety metric)", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(0.1, color="green", linestyle="--", alpha=0.6, label="medical safety target (<0.1)")
    ax2.legend(loc="lower left")
    for bar, v in zip(bars2, fnr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = PLOTS_DIR / "metric_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[+] saved {out}")


if __name__ == "__main__":
    plot_all_confusion_matrices()
    plot_pareto_scatter()
    plot_class_dist()
    plot_metric_comparison()
    print(f"\n[+] all plots saved to {PLOTS_DIR}/")
