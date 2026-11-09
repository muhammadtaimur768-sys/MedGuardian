"""Generate paper figures (paper/figures/*.pdf) from results/ and the system structure.

    python experiments/make_figures.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
INK, MUTED, G1, G2, G3, ENV = "#1f2933", "#52606d", "#2563eb", "#059669", "#d97706", "#7c3aed"


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", lw=1.4,
                                edgecolor=color, facecolor=color + "18"))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8.2, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.4, color=MUTED)


def arrow(ax, a, b, color=MUTED, text=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, lw=1.1, color=color))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.07, text, ha="center", fontsize=6, color=color)


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.2, 3.7, 2.0, 0.9, "Clinician task", "instruction + context", INK)
    box(ax, 3.0, 3.7, 2.6, 0.9, "LLM agent (Qwen2.5-7B)", "GET / POST / FINISH, 8 rounds", INK)
    box(ax, 6.6, 3.7, 3.2, 0.9, "FHIR R4 EHR (HAPI)", "MedAgentBench, 100 patients", ENV)
    box(ax, 3.0, 2.25, 2.6, 0.95, "G1 Typed tools + calculator", "filtered pages, age, windows", G1)
    box(ax, 6.6, 2.25, 3.2, 0.95, "G1 Pre-order verifier", "schema, codes, indication, dose; 1 repair", G1)
    box(ax, 0.2, 0.4, 3.0, 1.1, "G2 Process guardian", "timeline + BM25 guideline thresholds", G2)
    box(ax, 3.6, 0.4, 2.8, 1.1, "G3 Sonography fusion", "OCR calipers + Whisper dictation", G3)
    box(ax, 6.8, 0.4, 3.0, 1.1, "Trace + grader", "JSONL steps; base-paper refsol", MUTED)
    arrow(ax, (2.2, 4.15), (3.0, 4.15))
    arrow(ax, (5.6, 4.3), (6.6, 4.3), text="GET")
    arrow(ax, (6.6, 3.95), (5.6, 3.95), color=G1, text="summary")
    arrow(ax, (4.3, 3.7), (4.3, 3.2), color=G1, style="<|-|>")
    arrow(ax, (5.6, 2.9), (6.6, 2.9), color=G1, text="POST draft")
    arrow(ax, (8.2, 3.2), (8.2, 3.7), color=G1)
    ax.text(8.3, 3.42, "if valid", ha="left", va="center", fontsize=6, color=G1)
    arrow(ax, (8.3, 2.25), (8.3, 1.5), color=MUTED)
    arrow(ax, (1.2, 3.7), (1.7, 1.5), color=G2)
    arrow(ax, (1.2, 3.7), (4.6, 1.5), color=G3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "architecture.pdf"))
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=200)
    plt.close(fig)


def per_category():
    runs = {"E2 base-paper agent": "e2_basepaper_test50.csv", "E3 MedGuardian": "e3_medguardian_test50.csv"}
    data = {}
    for label, f in runs.items():
        p = os.path.join(ROOT, "results", f)
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            cats = ["task%d" % i for i in range(1, 11)]
            data[label] = [100.0 * sum(int(r["correct"]) for r in rows if r["category"] == c) /
                           max(1, sum(r["category"] == c for r in rows)) for c in cats]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    width = 0.8 / len(data)
    for k, (label, vals) in enumerate(data.items()):
        ax.bar([i + k * width for i in range(10)], vals, width, label=label, color=[MUTED, G1][k % 2])
    ax.set_xticks([i + width * (len(data) - 1) / 2 for i in range(10)])
    ax.set_xticklabels(["T%d" % i for i in range(1, 11)], fontsize=7)
    ax.set_ylabel("Success rate (%)", fontsize=7)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "per_category.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    architecture()
    per_category()
    print("wrote", sorted(os.listdir(OUT)))
