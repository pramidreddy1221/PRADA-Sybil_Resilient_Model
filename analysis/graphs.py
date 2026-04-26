import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
GRAPHS = os.path.join(BASE, "graphs")
os.makedirs(GRAPHS, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, fname):
    path = os.path.join(GRAPHS, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Graph 1 — Substitute Model Agreement per Round
# ---------------------------------------------------------------------------
def graph1():
    data = load("attack_performance.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "attacker_001":    ("Papernot (fixed lr=0.01, 10 epochs)", "steelblue", "o"),
        "attacker_cvsearch": ("CV-Search (lr=0.01, 160 epochs)",   "darkorange", "s"),
    }
    for key, (label, color, marker) in styles.items():
        rounds = [r["round"] for r in data[key]["rounds"]]
        agr    = [r["agreement"] * 100 for r in data[key]["rounds"]]
        ax.plot(rounds, agr, marker=marker, color=color, linewidth=2,
                markersize=7, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Agreement with Victim (%)")
    ax.set_title("Substitute Model Agreement per Round")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    save(fig, "01_attack_success.png")


# ---------------------------------------------------------------------------
# Graph 2 — Detection Rate vs Number of Sybil Accounts
# ---------------------------------------------------------------------------
def graph2():
    prada    = load("prada_n_sweep.json")
    js       = load("js_n_sweep.json")
    combined = load("combined_n_sweep.json")

    N_vals      = [r["N"] for r in prada]
    prada_dr    = [r["detection_pct"] for r in prada]
    js_dr       = [100.0 if r["detected"] else 0.0 for r in js]
    combined_dr = [100.0 if r["combined"] else 0.0 for r in combined]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(N_vals, prada_dr,    marker="o", color="crimson",    linewidth=2, markersize=7, label="PRADA only")
    ax.plot(N_vals, js_dr,       marker="s", color="steelblue",  linewidth=2, markersize=7, label="JS extension")
    ax.plot(N_vals, combined_dr, marker="^", color="seagreen",   linewidth=2, markersize=7, label="Combined")
    ax.axvline(x=64, color="gray", linestyle="--", linewidth=1.4, label="N=64 (PRADA fails)")

    ax.set_xlabel("Number of Sybil Accounts (N)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate vs Number of Sybil Accounts")
    ax.set_xticks(N_vals)
    ax.set_ylim(-8, 115)
    ax.legend()
    fig.tight_layout()
    save(fig, "02_sybil_n_sweep.png")


# ---------------------------------------------------------------------------
# Graph 3 — PRADA W Scores: Attackers vs Benign (combined bar chart)
# ---------------------------------------------------------------------------
def graph3():
    baseline = load("prada_baseline.json")
    attack   = load("attack_sweep.json")

    # Merge both files; attack_sweep wins on duplicates (more detail)
    accounts = {}
    for r in baseline:
        accounts[r["account_id"]] = {"W": r["W"], "flagged": r["flagged"]}
    for r in attack:
        accounts[r["account_id"]] = {"W": r["W"], "flagged": r["flagged"]}

    # Attackers first (flagged), then benign, alphabetical within each group
    sorted_items = sorted(accounts.items(), key=lambda x: (not x[1]["flagged"], x[0]))

    short = {
        "attacker_001":     "att_001\n(baseline)",
        "attacker_cvsearch":"att_cvsearch",
        "attacker_fgsm":    "att_fgsm",
        "attacker_ifgsm":   "att_ifgsm",
        "attacker_mifgsm":  "att_mifgsm",
        "benign_001":       "benign_001",
    }
    labels = [short.get(k, k) for k, _ in sorted_items]
    W      = [v["W"] for _, v in sorted_items]
    colors = ["crimson" if v["flagged"] else "seagreen" for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, W, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    ax.axhline(y=0.96, color="black", linestyle="--", linewidth=1.6)

    # Annotate W values above bars
    for bar, w in zip(bars, W):
        ax.text(bar.get_x() + bar.get_width() / 2, w + 0.001,
                f"{w:.4f}", ha="center", va="bottom", fontsize=8.5)

    legend_elements = [
        mpatches.Patch(color="crimson",  label="Attacker (flagged)"),
        mpatches.Patch(color="seagreen", label="Benign (clean)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="DELTA = 0.96"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_ylabel("Shapiro-Wilk W Score")
    ax.set_title("PRADA W Scores — Attackers vs Benign")
    ax.set_ylim(0.87, 1.01)
    fig.tight_layout()
    save(fig, "03_prada_baseline.png")


# ---------------------------------------------------------------------------
# Graph 4 — W Score vs FGSM Step Size (Lambda)
# ---------------------------------------------------------------------------
def graph4():
    data = load("lambda_attack_results.json")
    lambda_rows = [r for r in data if r["account_id"].startswith("lambda_")]

    labels  = [r["label"] for r in lambda_rows]
    W       = [r["W"] for r in lambda_rows]
    x       = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(x), W, marker="o", color="steelblue", linewidth=2, markersize=7)
    ax.axhline(y=0.96, color="crimson", linestyle="--", linewidth=1.6, label="DELTA = 0.96")

    for xi, (lbl, w) in enumerate(zip(labels, W)):
        ax.annotate(f"{w:.4f}", xy=(xi, w), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("FGSM Step Size (Lambda)")
    ax.set_ylabel("Shapiro-Wilk W Score")
    ax.set_title("W Score vs FGSM Step Size (Lambda)")
    ax.set_ylim(0.82, 1.0)
    ax.legend()
    fig.tight_layout()
    save(fig, "04_lambda_sweep.png")


# ---------------------------------------------------------------------------
# Graph 5 — JS Detection Robustness vs Normal Query Ratio
# ---------------------------------------------------------------------------
def graph5():
    data = load("mixed_sybil_ratios.json")

    ratios   = [int(r["ratio"] * 100) for r in data]
    within   = [r["within_js"] for r in data]
    cross    = [r["cross_js"]  for r in data]
    gap      = [r["cross_js"] - r["within_js"] for r in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ratios, within, marker="o", color="steelblue",  linewidth=2, markersize=7, label="Within-Sybil JS")
    ax.plot(ratios, cross,  marker="s", color="darkorange", linewidth=2, markersize=7, label="Sybil-Benign JS")
    ax.plot(ratios, gap,    marker="^", color="seagreen",   linewidth=2, markersize=7, label="Gap (cross − within)")
    ax.axhline(y=0.15, color="crimson", linestyle="--", linewidth=1.4, label="Threshold = 0.15")

    ax.set_xlabel("Normal Query Ratio (%)")
    ax.set_ylabel("JS Divergence")
    ax.set_title("JS Detection Robustness vs Normal Query Ratio")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r}%" for r in ratios])
    ax.legend()
    fig.tight_layout()
    save(fig, "05_mixed_sybil_robustness.png")


# ---------------------------------------------------------------------------
# Graph 6 — JS Detection vs False Positive Rate by Threshold
# ---------------------------------------------------------------------------
def graph6():
    data = load("js_threshold_sweep.json")

    pure  = sorted([r for r in data if r["source"] == "pure"],  key=lambda x: x["threshold"])
    mixed = sorted([r for r in data if r["source"] == "mixed"], key=lambda x: x["threshold"])

    thresholds = [r["threshold"] for r in pure]
    pure_det   = [100 if r["detected"] else 0 for r in pure]
    pure_fp    = [100 if r["FP"]       else 0 for r in pure]
    mixed_det  = [100 if r["detected"] else 0 for r in mixed]
    mixed_fp   = [100 if r["FP"]       else 0 for r in mixed]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, pure_det,  marker="o", color="steelblue",  linewidth=2, markersize=7,
            label="Pure Sybil — Detection Rate")
    ax.plot(thresholds, pure_fp,   marker="o", color="steelblue",  linewidth=2, markersize=7,
            linestyle="--", label="Pure Sybil — False Positive Rate")
    ax.plot(thresholds, mixed_det, marker="s", color="darkorange", linewidth=2, markersize=7,
            label="Mixed Sybil — Detection Rate")
    ax.plot(thresholds, mixed_fp,  marker="s", color="darkorange", linewidth=2, markersize=7,
            linestyle="--", label="Mixed Sybil — False Positive Rate")
    ax.axvline(x=0.15, color="gray", linestyle=":", linewidth=1.6, label="Threshold = 0.15 (chosen)")

    ax.set_xlabel("JS Threshold")
    ax.set_ylabel("Rate (%)")
    ax.set_title("JS Detection vs False Positive Rate by Threshold")
    ax.set_xticks(thresholds)
    ax.set_ylim(-8, 115)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "06_js_threshold_tradeoff.png")


# ---------------------------------------------------------------------------
# Graph 7 — Distance Metric Comparison for Sybil Detection
# ---------------------------------------------------------------------------
def graph7():
    data = load("metric_comparison.json")

    metrics = [r["metric"]      for r in data]
    within  = [r["within_mean"] for r in data]
    cross   = [r["cross_mean"]  for r in data]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, within, width, label="Within-Sybil",  color="steelblue",  edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, cross,  width, label="Sybil-Benign",  color="coral",       edgecolor="black", linewidth=0.5)

    # Annotate bar tops
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(cross),
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlabel("Distance Metric")
    ax.set_ylabel("Distance")
    ax.set_title("Distance Metric Comparison for Sybil Detection")
    ax.legend()
    fig.tight_layout()
    save(fig, "07_metric_comparison.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating graphs…")
    graph1()
    graph2()
    graph3()
    graph4()
    graph5()
    graph6()
    graph7()
    saved = [f for f in os.listdir(GRAPHS) if f.endswith(".png")]
    print(f"\nDone — {len(saved)} PNG files in {GRAPHS}:")
    for f in sorted(saved):
        print(f"  {f}")
