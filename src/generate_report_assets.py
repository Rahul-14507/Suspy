import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

OUT_DIR = "outputs"
DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
NAVY   = "#0D1B2A"
BLUE   = "#1B4F8A"
TEAL   = "#00B4D8"
CYAN   = "#90E0EF"
GREEN  = "#2DC653"
AMBER  = "#FFA500"
RED    = "#E63946"
LIGHT  = "#F1F5F9"
GREY   = "#64748B"

TIER_COLOURS = {
    "Critical": RED,
    "High":     AMBER,
    "Medium":   TEAL,
    "Low":      GREEN,
}

# ── Load artefacts ──────────────────────────────────────────────────────────
scores  = pd.read_csv(os.path.join(OUT_DIR, "risk_scores.csv"))
metrics = json.load(open(os.path.join(OUT_DIR, "metrics.json")))


# ── 1.  Stats banner ────────────────────────────────────────────────────────
def plot_stats_banner():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.patch.set_facecolor(NAVY)

    cards = [
        ("9,082",      "Accounts Scored",     TEAL),
        ("81",         "Mule Accounts\nFlagged (Critical)", RED),
        ("0.9916",     "Mean CV ROC-AUC",      GREEN),
        ("85.15%",     "Mean CV Recall\n(Threshold = 0.30)", AMBER),
    ]

    for ax, (val, label, colour) in zip(axes, cards):
        ax.set_facecolor(NAVY)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        # Card background
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02",
            linewidth=2, edgecolor=colour,
            facecolor="#1C2B3A"
        )
        ax.add_patch(rect)
        ax.text(0.50, 0.62, val,  ha="center", va="center",
                fontsize=30, fontweight="bold", color=colour,
                transform=ax.transAxes)
        ax.text(0.50, 0.25, label, ha="center", va="center",
                fontsize=11, color=LIGHT, transform=ax.transAxes,
                multialignment="center")

    fig.suptitle("MuleNet — Model Performance at a Glance",
                 fontsize=14, fontweight="bold", color=LIGHT, y=1.02)
    plt.tight_layout(pad=0.5)
    path = os.path.join(DOCS_DIR, "stats_banner.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved {path}")


# ── 2.  DRI distribution ────────────────────────────────────────────────────
def plot_dri_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(NAVY)

    # Left: histogram
    ax = axes[0]
    ax.set_facecolor("#1C2B3A")
    low_dri = scores[scores["DRI"] <= 30]["DRI"]
    high_dri = scores[scores["DRI"] > 30]["DRI"]
    ax.hist(low_dri,  bins=30, color=GREEN, alpha=0.85, label="Low risk   (DRI ≤ 30)")
    ax.hist(high_dri, bins=30, color=RED,   alpha=0.85, label="Critical   (DRI > 80)")
    ax.set_xlabel("Dynamic Risk Index (DRI)", color=LIGHT, fontsize=12)
    ax.set_ylabel("Number of Accounts",       color=LIGHT, fontsize=12)
    ax.set_title("DRI Score Distribution",    color=LIGHT, fontsize=13, fontweight="bold")
    ax.tick_params(colors=LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    ax.legend(facecolor="#1C2B3A", labelcolor=LIGHT, fontsize=10)

    # Right: pie / donut
    ax2 = axes[1]
    ax2.set_facecolor("#1C2B3A")
    tier_counts = scores["risk_tier"].value_counts()
    tier_order  = [t for t in ["Critical", "High", "Medium", "Low"] if t in tier_counts]
    sizes  = [tier_counts[t] for t in tier_order]
    colors = [TIER_COLOURS[t] for t in tier_order]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=tier_order, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(width=0.55, edgecolor=NAVY, linewidth=2),
        textprops=dict(color=LIGHT, fontsize=11),
    )
    for at in autotexts:
        at.set_color(NAVY)
        at.set_fontweight("bold")
    ax2.set_title("Risk Tier Distribution", color=LIGHT, fontsize=13, fontweight="bold")

    plt.tight_layout(pad=2)
    path = os.path.join(DOCS_DIR, "dri_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved {path}")


# ── 3.  Class imbalance ─────────────────────────────────────────────────────
def plot_class_imbalance():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor("#1C2B3A")

    classes = ["Clean Accounts (F3924 = 0)", "Mule Accounts (F3924 = 1)"]
    counts  = [9001, 81]
    colours = [TEAL, RED]

    bars = ax.barh(classes, counts, color=colours, height=0.5,
                   edgecolor=NAVY, linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({count/sum(counts)*100:.2f}%)",
                va="center", color=LIGHT, fontsize=11, fontweight="bold")

    ax.set_xlim(0, 10500)
    ax.set_xlabel("Number of Accounts", color=LIGHT, fontsize=12)
    ax.set_title("Dataset Class Imbalance  —  111 : 1 ratio",
                 color=LIGHT, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=LIGHT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Imbalance ratio annotation
    ax.text(0.98, 0.12, "Scale-pos-weight: 110.8×",
            transform=ax.transAxes, ha="right", fontsize=10,
            color=AMBER, style="italic")

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, "class_imbalance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved {path}")


# ── 4.  Architecture diagram ────────────────────────────────────────────────
def plot_architecture():
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_subplot(111)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis("off")

    fig.suptitle("MuleNet — Full Architecture Vision",
                 fontsize=15, fontweight="bold", color=LIGHT, y=0.97)

    layers = [
        # (x_centre, label_top, label_bottom, colour, is_mvp)
        (1.2,  "LAYER 1",          "Feature\nEngineering",       TEAL,  True),
        (3.7,  "LAYER 2a",         "XGBoost\nClassifier + SHAP", BLUE,  True),
        (6.5,  "LAYER 2b\n(Next)", "GraphSAGE\nGNN",             GREY,  False),
        (9.2,  "LAYER 3\n(Next)",  "DRI Fusion\nMeta-Learner",   GREY,  False),
        (11.8, "FUTURE",           "Investigator\nDashboard",     GREY,  False),
    ]

    for x, top, bot, colour, is_mvp in layers:
        alpha = 1.0 if is_mvp else 0.45
        # Box
        rect = mpatches.FancyBboxPatch(
            (x - 0.95, 1.3), 1.9, 2.4,
            boxstyle="round,pad=0.1",
            linewidth=2.5 if is_mvp else 1,
            edgecolor=colour,
            facecolor="#1C2B3A" if is_mvp else "#141E29",
            alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(x, 3.3, top, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color=colour if is_mvp else GREY,
                alpha=alpha)
        ax.text(x, 2.2, bot, ha="center", va="center",
                fontsize=10, color=LIGHT, alpha=alpha,
                multialignment="center")

        if is_mvp:
            badge = mpatches.FancyBboxPatch(
                (x - 0.55, 0.85), 1.10, 0.35,
                boxstyle="round,pad=0.04",
                facecolor=colour, edgecolor="none"
            )
            ax.add_patch(badge)
            ax.text(x, 1.025, "MVP ✓", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=NAVY)

    # Arrows
    arrow_xs = [(2.15, 2.75), (4.65, 5.55), (7.45, 8.25), (10.15, 10.85)]
    for x0, x1 in arrow_xs:
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5),
                    arrowprops=dict(arrowstyle="->", color=GREY,
                                    lw=1.5, connectionstyle="arc3,rad=0"))

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, "architecture.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved {path}")


# ── 5. Top features bar chart ───────────────────────────────────────────────
def plot_top_features_styled():
    """Styled version of top features using gain from the saved model."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(OUT_DIR, "mule_classifier.json"))
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [s[0] for s in sorted_imp][::-1]
    gains = [s[1] for s in sorted_imp][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor("#1C2B3A")

    bar_colours = [TEAL if g < max(gains) * 0.5 else BLUE if g < max(gains) * 0.8 else CYAN
                   for g in gains]
    bars = ax.barh(names, gains, color=bar_colours, edgecolor=NAVY, linewidth=0.8, height=0.65)

    for bar, gain in zip(bars, gains):
        ax.text(bar.get_width() + max(gains) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{gain:.0f}", va="center", color=LIGHT, fontsize=9)

    ax.set_xlabel("Information Gain", color=LIGHT, fontsize=12)
    ax.set_title("Top 15 Features  —  XGBoost (Gain)",
                 color=LIGHT, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=LIGHT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, "top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved {path}")


# ── Run all ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating report assets...")
    plot_stats_banner()
    plot_dri_distribution()
    plot_class_imbalance()
    plot_architecture()
    plot_top_features_styled()
    print("\nAll assets saved to docs/")
