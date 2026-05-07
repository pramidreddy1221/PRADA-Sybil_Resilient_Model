import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from defense.logs import load_logs
from defense.distances import compute_dmin_per_account
from defense.sybil_detection import build_histograms, compute_pairwise_js
from config import SYBIL_MIN_DMIN, SYBIL_JS_THRESHOLD, SYBIL_N_BINS, LOG_PATH

N_SYBIL = 64
N_THRESHOLDS = 500
ATTACKER_LIMIT = 6400
PARTIAL_AUC_FPR_MAX = 0.10


def redistribute(records, n_accounts, prefix):
    result = []
    for i, rec in enumerate(records):
        new_rec = dict(rec)
        new_rec["account_id"] = f"{prefix}{(i % n_accounts) + 1:03d}"
        result.append(new_rec)
    return result


def build_mixed_accounts(attacker_pool, benign_pool, n_accounts, prefix):
    result = []
    atk_idx = 0
    ben_idx = 0
    for i in range(n_accounts):
        account_id = f"{prefix}{i + 1:03d}"
        assigned = 0
        while assigned < 100:
            for _ in range(7):
                if assigned >= 100:
                    break
                rec = dict(benign_pool[ben_idx % len(benign_pool)])
                rec["account_id"] = account_id
                result.append(rec)
                ben_idx += 1
                assigned += 1
            for _ in range(3):
                if assigned >= 100:
                    break
                rec = dict(attacker_pool[atk_idx % len(attacker_pool)])
                rec["account_id"] = account_id
                result.append(rec)
                atk_idx += 1
                assigned += 1
    return result


def compute_confusion(scores_arr, labels_arr, thresh, pos_total, neg_total):
    pred = scores_arr < thresh
    tp = int(np.sum(pred & (labels_arr == 1)))
    fp = int(np.sum(pred & (labels_arr == 0)))
    fn = pos_total - tp
    tn = neg_total - fp
    tpr = tp / pos_total if pos_total > 0 else 0.0
    fpr = fp / neg_total if neg_total > 0 else 0.0
    return tp, fp, fn, tn, tpr, fpr


def pair_stats(a):
    if len(a) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(a),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def run_analysis(name, sybil_records, benign_baseline_records, n_benign_total):
    all_recs = sybil_records + benign_baseline_records
    account_dmin_data = compute_dmin_per_account(all_recs)
    eligible = {
        acct: data["D"]
        for acct, data in account_dmin_data.items()
        if len(data["D"]) >= SYBIL_MIN_DMIN
    }

    n_sybil_elig = sum(1 for a in eligible if not a.startswith("benign_"))
    n_benign_elig = sum(1 for a in eligible if a.startswith("benign_"))

    print(f"\n{name}")
    print(f"eligible sybil accounts: {n_sybil_elig}")
    print(f"eligible benign accounts: {n_benign_elig}")

    if n_sybil_elig < 0.8 * N_SYBIL:
        print(f"WARNING: eligible sybil {n_sybil_elig} below 80% of {N_SYBIL}")
    if n_benign_elig < 0.8 * n_benign_total:
        print(f"WARNING: eligible benign {n_benign_elig} below 80% of {n_benign_total}")

    histograms = build_histograms(eligible, SYBIL_N_BINS)
    accounts, js_matrix = compute_pairwise_js(histograms)
    n_acc = len(accounts)

    # ROC is pair-level. Sybil-Sybil pairs are positive (low JS = coordinated).
    # Benign-benign pairs are negative (high JS = independent). Sybil-benign pairs
    # are excluded from labeling.
    ss, bb, sb = [], [], []
    roc_scores, roc_labels = [], []

    for i in range(n_acc):
        for j in range(i + 1, n_acc):
            bi = accounts[i].startswith("benign_")
            bj = accounts[j].startswith("benign_")
            v = float(js_matrix[i, j])
            if not bi and not bj:
                ss.append(v)
                roc_scores.append(v)
                roc_labels.append(1)
            elif bi and bj:
                bb.append(v)
                roc_scores.append(v)
                roc_labels.append(0)
            else:
                sb.append(v)

    ss_arr = np.array(ss)
    bb_arr = np.array(bb)
    sb_arr = np.array(sb)
    scores_arr = np.array(roc_scores)
    labels_arr = np.array(roc_labels)

    pos_total = int(np.sum(labels_arr == 1))
    neg_total = int(np.sum(labels_arr == 0))

    print(f"SS pairs: {len(ss)}, BB pairs: {len(bb)}, SB pairs: {len(sb)}")
    print(f"total positive: {pos_total}, total negative: {neg_total}")

    all_vals = np.concatenate([ss_arr, bb_arr, sb_arr]) if len(ss) + len(bb) + len(sb) > 0 else np.array([0.0])
    max_js = float(np.max(all_vals))
    thresh_max = max_js + 0.01

    thresholds = np.linspace(0.0, thresh_max, N_THRESHOLDS)
    tprs = np.zeros(N_THRESHOLDS)
    fprs = np.zeros(N_THRESHOLDS)

    for idx, t in enumerate(thresholds):
        pred = scores_arr < t
        tp = int(np.sum(pred & (labels_arr == 1)))
        fp = int(np.sum(pred & (labels_arr == 0)))
        tprs[idx] = tp / pos_total if pos_total > 0 else 0.0
        fprs[idx] = fp / neg_total if neg_total > 0 else 0.0

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

    y_tp, y_fp, y_fn, y_tn, y_tpr2, y_fpr2 = compute_confusion(scores_arr, labels_arr, yt, pos_total, neg_total)
    s_tp, s_fp, s_fn, s_tn, s_tpr, s_fpr = compute_confusion(scores_arr, labels_arr, SYBIL_JS_THRESHOLD, pos_total, neg_total)

    print(f"Youden CM: TP={y_tp}, FP={y_fp}, FN={y_fn}, TN={y_tn}, TPR={y_tpr2:.4f}, FPR={y_fpr2:.4f}")
    print(f"sweep CM: TP={s_tp}, FP={s_fp}, FN={s_fn}, TN={s_tn}, TPR={s_tpr:.4f}, FPR={s_fpr:.4f}")

    diff = abs(yt - SYBIL_JS_THRESHOLD)
    validated = diff < 0.05
    if validated:
        print(f"sweep threshold {SYBIL_JS_THRESHOLD} validated (|diff|={diff:.4f} < 0.05)")
    else:
        print(f"sweep threshold {SYBIL_JS_THRESHOLD} discrepant from Youden={yt:.4f} (|diff|={diff:.4f})")

    sweep_in = bool(safe_lo is not None and safe_lo <= SYBIL_JS_THRESHOLD <= safe_hi)
    youden_in = bool(safe_lo is not None and safe_lo <= yt <= safe_hi)

    result = {
        "auc": auc,
        "partial_auc": partial_auc,
        "partial_auc_fpr_max": PARTIAL_AUC_FPR_MAX,
        "pair_counts": {
            "sybil_sybil": len(ss),
            "benign_benign": len(bb),
            "sybil_benign": len(sb),
            "total_positive": pos_total,
            "total_negative": neg_total,
        },
        "pair_stats": {
            "sybil_sybil": pair_stats(ss_arr),
            "benign_benign": pair_stats(bb_arr),
            "sybil_benign": pair_stats(sb_arr),
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
            "threshold": SYBIL_JS_THRESHOLD,
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

    viz = {
        "ss_arr": ss_arr,
        "bb_arr": bb_arr,
        "sb_arr": sb_arr,
        "thresholds": thresholds,
        "tprs": tprs,
        "fprs": fprs,
        "sorted_fprs": sorted_fprs,
        "sorted_tprs": sorted_tprs,
        "max_js": max_js,
        "thresh_max": thresh_max,
        "yt": yt,
        "y_tpr": y_tpr,
        "y_fpr": y_fpr,
        "safe_lo": safe_lo,
        "safe_hi": safe_hi,
        "safe_mask": safe_mask,
        "auc": auc,
        "partial_auc": partial_auc,
        "y_tp": y_tp,
        "y_fp": y_fp,
        "y_fn": y_fn,
        "y_tn": y_tn,
        "s_tpr": s_tpr,
        "s_fpr": s_fpr,
        "validated": validated,
    }

    return result, viz


def plot_hist(ax, viz, title):
    ss_arr = viz["ss_arr"]
    bb_arr = viz["bb_arr"]
    sb_arr = viz["sb_arr"]
    thresh_max = viz["thresh_max"]
    yt = viz["yt"]
    safe_lo = viz["safe_lo"]
    safe_hi = viz["safe_hi"]

    bins = np.linspace(0.0, thresh_max, 65)

    ax.hist(ss_arr, bins=bins, density=True, alpha=0.6, color="steelblue", label="Sybil-Sybil")
    ax.hist(sb_arr, bins=bins, density=True, alpha=0.6, color="darkorange", label="Sybil-Benign")
    ax.hist(bb_arr, bins=bins, density=True, alpha=0.6, color="forestgreen", label="Benign-Benign")

    if safe_lo is not None and safe_hi is not None:
        ax.axvspan(0.0, safe_lo, alpha=0.08, color="red")
        ax.axvspan(safe_hi, thresh_max, alpha=0.08, color="orange")
        ax.axvspan(safe_lo, safe_hi, alpha=0.15, color="mediumpurple",
                   label=f"safe [{safe_lo:.3f}, {safe_hi:.3f}]")

    ax.axvline(yt, color="black", linestyle="--", linewidth=1.5,
               label=f"Youden τ={yt:.3f}")
    ax.axvline(SYBIL_JS_THRESHOLD, color="grey", linestyle=":", linewidth=1.5,
               label=f"sweep τ={SYBIL_JS_THRESHOLD}")

    ax.set_xlabel("JS Divergence")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0.0, thresh_max)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)


def plot_roc(ax, viz):
    sorted_fprs = viz["sorted_fprs"]
    sorted_tprs = viz["sorted_tprs"]
    thresholds = viz["thresholds"]
    tprs = viz["tprs"]
    fprs = viz["fprs"]
    auc = viz["auc"]
    partial_auc = viz["partial_auc"]
    yt = viz["yt"]
    y_tpr = viz["y_tpr"]
    y_fpr = viz["y_fpr"]
    s_tpr = viz["s_tpr"]
    s_fpr = viz["s_fpr"]
    safe_lo = viz["safe_lo"]
    safe_hi = viz["safe_hi"]
    safe_mask = viz["safe_mask"]
    y_tp = viz["y_tp"]
    y_fp = viz["y_fp"]
    y_fn = viz["y_fn"]
    y_tn = viz["y_tn"]

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.axvspan(0, PARTIAL_AUC_FPR_MAX, alpha=0.08, color="lightblue",
               label=f"pAUC={partial_auc:.4f}")

    if safe_lo is not None and safe_hi is not None:
        arc_fpr = fprs[safe_mask]
        arc_tpr = tprs[safe_mask]
        if len(arc_fpr) > 1:
            arc_sort = np.argsort(arc_fpr)
            ax.plot(arc_fpr[arc_sort], arc_tpr[arc_sort], color="gold", linewidth=6,
                    alpha=0.3, label=f"safe arc [{safe_lo:.3f}, {safe_hi:.3f}]")

    ax.plot(sorted_fprs, sorted_tprs, color="steelblue", linewidth=2,
            label=f"AUC={auc:.4f}")

    ax.scatter([y_fpr], [y_tpr], color="gold", marker="*", s=225, zorder=5,
               label=f"Youden τ={yt:.3f} TPR={y_tpr:.3f} FPR={y_fpr:.3f}")
    ax.scatter([s_fpr], [s_tpr], color="red", marker="o", s=100, zorder=5,
               label=f"sweep τ={SYBIL_JS_THRESHOLD} TPR={s_tpr:.3f} FPR={s_fpr:.3f}")

    cm_text = f"TP={y_tp}\nFP={y_fp}\nFN={y_fn}\nTN={y_tn}"
    ax.text(0.98, 0.35, cm_text, transform=ax.transAxes, fontsize=8,
            va="center", ha="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="steelblue"))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"AUC={auc:.4f} | pAUC={partial_auc:.4f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    all_records = load_logs()
    print(f"loaded {len(all_records)} records from {LOG_PATH}")

    attacker_pool = [r for r in all_records if r["account_id"] == "attacker_001"][:ATTACKER_LIMIT]
    benign_pool = [r for r in all_records if r["account_id"] == "benign_001"]
    print(f"attacker_001: {len(attacker_pool)} records")
    print(f"benign_001: {len(benign_pool)} records")

    n_benign = len(benign_pool) // 100
    print(f"N_BENIGN: {n_benign}")

    benign_baseline = redistribute(benign_pool[:n_benign * 100], n_benign, "benign_")

    rr_sybil = redistribute(attacker_pool, N_SYBIL, "sybil_rr_")

    rand_pool = list(attacker_pool)
    random.seed(42)
    random.shuffle(rand_pool)
    rand_sybil = redistribute(rand_pool, N_SYBIL, "sybil_rand_")

    mix_sybil = build_mixed_accounts(attacker_pool, benign_pool, N_SYBIL, "sybil_mix_")

    dist_configs = [
        ("round_robin", "Round-Robin", rr_sybil),
        ("randomized", "Randomized", rand_sybil),
        ("mixed_70_30", "Mixed 70-30", mix_sybil),
    ]

    all_results = {}
    all_viz = {}
    summary_rows = []

    for key, display_name, sybil_recs in dist_configs:
        result, viz = run_analysis(display_name, sybil_recs, benign_baseline, n_benign)
        all_results[key] = result
        all_viz[key] = viz
        validated_str = "yes" if abs(result["youden"]["threshold"] - SYBIL_JS_THRESHOLD) < 0.05 else "no"
        summary_rows.append((display_name, result["auc"], result["partial_auc"],
                              result["youden"]["threshold"], result["youden"]["tpr"],
                              result["youden"]["fpr"], validated_str))

    print("\nDistribution           AUC     Partial AUC   Youden T   TPR      FPR      Validated")
    for row in summary_rows:
        name, auc, pauc, yt, tpr, fpr, val = row
        print(f"{name:<22} {auc:.4f}  {pauc:.4f}        {yt:.4f}   {tpr:.4f}   {fpr:.4f}   {val}")

    out_json = ROOT / "analysis" / "results" / "roc_distributions.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_json}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    fig.suptitle("ROC Analysis — JS Sybil Detector Across Query Distributions",
                 fontsize=14, fontweight="bold")

    for col, (key, display_name, _) in enumerate(dist_configs):
        plot_hist(axes[0, col], all_viz[key], display_name)
        plot_roc(axes[1, col], all_viz[key])

    plt.tight_layout()
    out_img = ROOT / "analysis" / "graphs" / "09_roc_distributions.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Saved → {out_img}")


if __name__ == "__main__":
    main()
