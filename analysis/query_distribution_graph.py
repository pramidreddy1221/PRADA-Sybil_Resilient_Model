import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYBIL_JS_THRESHOLD = 0.15

DIST_KEY_MAP = {
    "round-robin": "round_robin",
    "randomized": "randomized",
    "mixed-70-30": "mixed_70_30",
}
LABELS = ["Round-Robin", "Randomized", "Mixed 70-30"]


def bar_color(gap):
    if gap > 0.15:
        return "forestgreen"
    if gap > 0.10:
        return "steelblue"
    return "darkorange"


def main():
    sweep_path = ROOT / "simulation" / "query_distribution_results.json"
    roc_path = ROOT / "analysis" / "results" / "roc_distributions.json"

    sweep_data = json.loads(sweep_path.read_text(encoding="utf-8"))
    roc_data = json.loads(roc_path.read_text(encoding="utf-8"))

    rows = []
    for entry in sweep_data:
        key = DIST_KEY_MAP[entry["name"]]
        roc = roc_data[key]
        within_js = roc["pair_stats"]["sybil_sybil"]["mean"]
        cross_js = roc["pair_stats"]["sybil_benign"]["mean"]
        gap = cross_js - within_js
        detected = entry["detected"]
        rows.append({
            "name": entry["name"],
            "within_js": within_js,
            "cross_js": cross_js,
            "gap": gap,
            "detected": detected,
        })

    for r in rows:
        print(f"{r['name']}: within_js={r['within_js']:.4f}, cross_js={r['cross_js']:.4f}, gap={r['gap']:.4f}, detected={r['detected']}")

    within_vals = np.array([r["within_js"] for r in rows])
    cross_vals = np.array([r["cross_js"] for r in rows])
    gap_vals = np.array([r["gap"] for r in rows])
    detected_flags = [r["detected"] for r in rows]

    x = np.arange(len(LABELS))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
    fig.suptitle("Query Distribution Impact on JS Sybil Detection", fontsize=14, fontweight="bold")

    bars_within = ax1.bar(x - w / 2, within_vals, w, color="steelblue", label="Within-JS (Sybil-Sybil)")
    bars_cross = ax1.bar(x + w / 2, cross_vals, w, color="darkorange", label="Cross-JS (Sybil-Benign)")

    for bar in bars_within:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_cross:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax1.axhline(SYBIL_JS_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
                label=f"Detection threshold ({SYBIL_JS_THRESHOLD})")

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("Mean JS Divergence")
    ax1.set_title("Within-Sybil vs Cross-Group JS Divergence")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)

    colors = [bar_color(g) for g in gap_vals]
    bars_gap = ax2.bar(x, gap_vals, 0.5, color=colors)

    for i, (bar, gap, det) in enumerate(zip(bars_gap, gap_vals, detected_flags)):
        label_y = bar.get_height() + 0.003 if gap >= 0 else gap - 0.012
        ax2.text(bar.get_x() + bar.get_width() / 2, label_y,
                 f"{gap:.3f}", ha="center", va="bottom", fontsize=8)
        det_label = "Detected" if det else "Not Detected"
        det_y = label_y + 0.018 if gap >= 0 else label_y - 0.018
        ax2.text(bar.get_x() + bar.get_width() / 2, det_y,
                 det_label, ha="center", va="bottom", fontsize=8)

    ax2.axhline(0, color="grey", linestyle="--", linewidth=1.0)

    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("JS Divergence Gap (Cross - Within)")
    ax2.set_title("Detection Gap Across Query Distributions")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    out_path = ROOT / "analysis" / "graphs" / "11_query_distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
