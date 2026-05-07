import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from defense.logs import load_logs
from defense.prada import run_prada_on_records
from config import LOG_PATH, DELTA, MIN_QUERIES

N_THRESHOLDS = 500
PARTIAL_AUC_FPR_MAX = 0.10
JS_AUC = 0.9978


def compute_confusion(w_scores, labels, thresh, pos_total, neg_total):
    pred = np.array(w_scores) < thresh
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = pos_total - tp
    tn = neg_total - fp
    tpr = tp / pos_total if pos_total > 0 else 0.0
    fpr = fp / neg_total if neg_total > 0 else 0.0
    return tp, fp, fn, tn, tpr, fpr


def main():
    all_records = load_logs()

    attacker_records = [r for r in all_records if r["account_id"] == "attacker_001"]
    benign_records = [r for r in all_records if r["account_id"] == "benign_001"]
    print(f"attacker_001: {len(attacker_records)} records")
    print(f"benign_001: {len(benign_records)} records")

    prada_results = run_prada_on_records(attacker_records + benign_records, DELTA)

    w_attacker = prada_results["attacker_001"]["W"]
    w_benign = prada_results["benign_001"]["W"]
    print(f"attacker_001 W: {w_attacker}")
    print(f"benign_001 W: {w_benign}")

    # AUC computed from 2 data points only (1 attacker, 1 benign) — result is a step function, not a robust estimate.
    w_scores = np.array([w_attacker, w_benign], dtype=float)
    labels = np.array([1, 0])
    pos_total = 1
    neg_total = 1

    thresholds = np.linspace(0.0, 1.0, N_THRESHOLDS)
    tprs = np.zeros(N_THRESHOLDS)
    fprs = np.zeros(N_THRESHOLDS)

    for idx, t in enumerate(thresholds):
        pred = w_scores < t
        tprs[idx] = float(np.sum(pred & (labels == 1))) / pos_total
        fprs[idx] = float(np.sum(pred & (labels == 0))) / neg_total

    sort_idx = np.argsort(fprs)
    sorted_fprs = fprs[sort_idx]
    sorted_tprs = tprs[sort_idx]
    sorted_thresholds = thresholds[sort_idx]
    auc = float(np.trapezoid(sorted_tprs, sorted_fprs))

    pmask = sorted_fprs <= PARTIAL_AUC_FPR_MAX
    if np.sum(pmask) >= 1:
        pfprs = sorted_fprs[pmask].copy()
        ptprs = sorted_tprs[pmask].copy()
        next_i = int(np.sum(pmask))
        if pfprs[-1] < PARTIAL_AUC_FPR_MAX and next_i < len(sorted_fprs):
            x1, x2 = sorted_fprs[next_i - 1], sorted_fprs[next_i]
            y1, y2 = sorted_tprs[next_i - 1], sorted_tprs[next_i]
            tpr_bnd = y1 + (y2 - y1) * (PARTIAL_AUC_FPR_MAX - x1) / (x2 - x1) if x2 > x1 else y2
            pfprs = np.append(pfprs, PARTIAL_AUC_FPR_MAX)
            ptprs = np.append(ptprs, tpr_bnd)
        partial_auc = float(np.trapezoid(ptprs, pfprs))
    else:
        partial_auc = 0.0

    print(f"AUC: {auc:.4f}")
    print(f"partial AUC (FPR<=0.10): {partial_auc:.4f}")

    j_vals = tprs - fprs
    best_i = int(np.argmax(j_vals))
    yt = float(thresholds[best_i])
    y_tpr = float(tprs[best_i])
    y_fpr = float(fprs[best_i])
    y_j = float(j_vals[best_i])
    print(f"Youden threshold: {yt:.4f}, TPR={y_tpr:.4f}, FPR={y_fpr:.4f}")

    safe_mask = (tprs >= 0.90) & (fprs <= 0.10)
    safe_thresh = thresholds[safe_mask]
    if len(safe_thresh) > 0:
        safe_lo = float(safe_thresh.min())
        safe_hi = float(safe_thresh.max())
        print(f"safe range: [{safe_lo:.4f}, {safe_hi:.4f}]")
    else:
        safe_lo = None
        safe_hi = None
        print("safe range: none")

    y_tp, y_fp, y_fn, y_tn, y_tpr2, y_fpr2 = compute_confusion(w_scores, labels, yt, pos_total, neg_total)
    s_tp, s_fp, s_fn, s_tn, s_tpr, s_fpr = compute_confusion(w_scores, labels, DELTA, pos_total, neg_total)

    print(f"Youden CM: TP={y_tp}, FP={y_fp}, FN={y_fn}, TN={y_tn}, TPR={y_tpr2:.4f}, FPR={y_fpr2:.4f}")
    print(f"DELTA={DELTA} CM: TP={s_tp}, FP={s_fp}, FN={s_fn}, TN={s_tn}, TPR={s_tpr:.4f}, FPR={s_fpr:.4f}")

    diff = abs(yt - DELTA)
    validated = diff < 0.05
    if validated:
        print(f"DELTA={DELTA} validated (|diff|={diff:.4f} < 0.05)")
    else:
        print(f"DELTA={DELTA} discrepant from Youden={yt:.4f} (|diff|={diff:.4f})")

    sweep_in = bool(safe_lo is not None and safe_lo <= DELTA <= safe_hi)
    youden_in = bool(safe_lo is not None and safe_lo <= yt <= safe_hi)

    result = {
        "auc": auc,
        "partial_auc": partial_auc,
        "partial_auc_fpr_max": PARTIAL_AUC_FPR_MAX,
        "w_scores": {
            "attacker_001": w_attacker,
            "benign_001": w_benign,
        },
        "youden": {
            "threshold": yt,
            "tpr": y_tpr,
            "fpr": y_fpr,
            "j_score": y_j,
            "confusion": {
                "tp": y_tp, "fp": y_fp, "fn": y_fn, "tn": y_tn,
                "tpr": y_tpr2, "fpr": y_fpr2,
            },
        },
        "sweep": {
            "threshold": DELTA,
            "tpr": s_tpr,
            "fpr": s_fpr,
            "confusion": {
                "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
                "tpr": s_tpr, "fpr": s_fpr,
            },
        },
        "safe_range": {
            "low": safe_lo,
            "high": safe_hi,
            "tpr_min": 0.90,
            "fpr_max": 0.10,
            "sweep_in_range": sweep_in,
            "youden_in_range": youden_in,
        },
        "roc_curve": {
            "fpr": [round(x, 6) for x in sorted_fprs.tolist()],
            "tpr": [round(x, 6) for x in sorted_tprs.tolist()],
            "thresholds": [round(x, 6) for x in sorted_thresholds.tolist()],
        },
    }

    out_json = ROOT / "analysis" / "results" / "prada_roc.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved → {out_json}")

    diff_auc = auc - JS_AUC
    if diff_auc > 0:
        print(f"PRADA AUC={auc:.4f} vs JS AUC={JS_AUC:.4f}: PRADA better by {diff_auc:.4f}")
    elif diff_auc < 0:
        print(f"PRADA AUC={auc:.4f} vs JS AUC={JS_AUC:.4f}: JS better by {abs(diff_auc):.4f}")
    else:
        print(f"PRADA AUC={auc:.4f} vs JS AUC={JS_AUC:.4f}: equal")

    bins = np.linspace(0.0, 1.0, 65)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
    fig.suptitle("ROC Analysis — PRADA Shapiro-Wilk Detector", fontsize=13, fontweight="bold")

    ax1.hist([w_attacker], bins=bins, density=True, alpha=0.6, color="red", label="attacker_001")
    ax1.hist([w_benign], bins=bins, density=True, alpha=0.6, color="forestgreen", label="benign_001")

    if safe_lo is not None and safe_hi is not None:
        ax1.axvspan(0.0, safe_lo, alpha=0.08, color="red")
        ax1.axvspan(safe_hi, 1.0, alpha=0.08, color="orange")
        ax1.axvspan(safe_lo, safe_hi, alpha=0.15, color="mediumpurple",
                    label=f"safe [{safe_lo:.3f}, {safe_hi:.3f}]")

    ax1.axvline(yt, color="black", linestyle="--", linewidth=1.5,
                label=f"Youden τ={yt:.3f}")
    ax1.axvline(DELTA, color="grey", linestyle=":", linewidth=1.5,
                label=f"DELTA={DELTA}")

    ax1.set_xlabel("Shapiro-Wilk W Score")
    ax1.set_ylabel("Density")
    ax1.set_title("W Score Distributions")
    ax1.set_xlim(0.0, 1.0)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax2.axvspan(0, PARTIAL_AUC_FPR_MAX, alpha=0.08, color="lightblue",
                label=f"pAUC={partial_auc:.4f}")

    if safe_lo is not None and safe_hi is not None:
        arc_fpr = fprs[safe_mask]
        arc_tpr = tprs[safe_mask]
        if len(arc_fpr) > 1:
            arc_sort = np.argsort(arc_fpr)
            ax2.plot(arc_fpr[arc_sort], arc_tpr[arc_sort], color="gold", linewidth=6,
                     alpha=0.3, label=f"safe arc [{safe_lo:.3f}, {safe_hi:.3f}]")

    ax2.plot(sorted_fprs, sorted_tprs, color="steelblue", linewidth=2,
             label=f"AUC={auc:.4f}")

    ax2.scatter([y_fpr], [y_tpr], color="gold", marker="*", s=225, zorder=5,
                label=f"Youden τ={yt:.3f} TPR={y_tpr:.3f} FPR={y_fpr:.3f}")
    ax2.scatter([s_fpr], [s_tpr], color="red", marker="o", s=100, zorder=5,
                label=f"DELTA={DELTA} TPR={s_tpr:.3f} FPR={s_fpr:.3f}")

    cm_text = f"TP={y_tp}\nFP={y_fp}\nFN={y_fn}\nTN={y_tn}"
    ax2.text(0.98, 0.35, cm_text, transform=ax2.transAxes, fontsize=8,
             va="center", ha="right", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="steelblue"))

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"AUC={auc:.4f} | pAUC={partial_auc:.4f}")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_img = ROOT / "analysis" / "graphs" / "10_prada_roc.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Saved → {out_img}")


if __name__ == "__main__":
    main()
