import sys
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defense.logs import load_logs
from defense.prada import run_prada_on_records
from defense.sybil_detection import run_sybil_detection, compute_pairwise_js, build_histograms
from defense.distances import compute_dmin_per_account
from simulation.sybil import redistribute_queries
from config import LOG_PATH, SYBIL_MIN_DMIN, SYBIL_JS_THRESHOLD


def prada_summary(prada_results: dict, prefix: str = "") -> tuple:
    accounts = [a for a in prada_results if a.startswith(prefix)] if prefix else list(prada_results.keys())
    n_flagged = sum(1 for a in accounts if prada_results[a]["flagged"])
    n_warmup = sum(1 for a in accounts if prada_results[a]["W"] is None)
    return n_flagged, n_warmup, accounts


def evaluate():
    print("  COMBINED EVALUATION: PRADA + Sybil Detection")
    print("  Extending PRADA for Sybil-Resilient Model Extraction Detection")

    all_records = load_logs(LOG_PATH)
    attacker_records = [r for r in all_records if r["account_id"] == "attacker_001"][:6400]
    benign_records = [r for r in all_records if r["account_id"] == "benign_001"][:3000]

    print(f"\nLog: {LOG_PATH.name}")
    print(f"  attacker_001: {len(attacker_records):>5} queries")
    print(f"  benign_001: {len(benign_records):>5} queries")

    summary_rows = []

    _section("SCENARIO 0: Baseline — single attacker_001 (N=1)")

    prada_res = run_prada_on_records(attacker_records)
    r0 = prada_res["attacker_001"]
    prada_str = f"YES (W={r0['W']})" if r0["flagged"] else f"NO (W={r0['W']})"
    print(f"  PRADA: {prada_str}")
    print(f"  Sybil detect: N/A (single account — cross-account analysis requires >= 3)")
    combined_detected = r0["flagged"]
    summary_rows.append(_row("N=1 baseline", prada_str, "N/A", combined_detected, is_attack=True))

    for N in [2, 5, 10, 64]:
        _section(f"SCENARIO: Sybil N={N}  (~{len(attacker_records)//N} queries/account)")

        sybil_records = redistribute_queries(attacker_records, N)

        prada_res = run_prada_on_records(sybil_records)
        n_prada_flagged, n_warmup, _ = prada_summary(prada_res)
        prada_str = (
            f"YES ({n_prada_flagged}/{N})" if n_prada_flagged > 0
            else f"NO  (0/{N} — {n_warmup} in warmup)"
        )
        print(f"  PRADA: {prada_str}")

        print(f"  Sybil detect:")
        sd_res = run_sybil_detection(sybil_records, verbose=True)

        sybil_str = (
            f"YES ({sd_res['cluster_size']} accounts flagged)"
            if sd_res["sybil_detected"]
            else f"NO  ({sd_res['reason']})"
        )

        combined_detected = (n_prada_flagged > 0) or sd_res["sybil_detected"]
        summary_rows.append(_row(f"N={N} Sybil", prada_str, sybil_str, combined_detected, is_attack=True))

    _section("SCENARIO: Benign only — benign_001")

    prada_res = run_prada_on_records(benign_records)
    r_b = prada_res.get("benign_001", {})
    n_b_flagged = int(r_b.get("flagged", False))
    prada_str = f"{'YES (FP!)' if n_b_flagged else 'NO '} (W={r_b.get('W','?')})"
    print(f"  PRADA: {prada_str}")

    print(f"  Sybil detect:")
    sd_benign = run_sybil_detection(benign_records, verbose=True)
    sybil_str = "NO " if not sd_benign["sybil_detected"] else "YES (FP!)"

    combined_fp = bool(n_b_flagged) or sd_benign["sybil_detected"]
    summary_rows.append(_row("Benign only", prada_str, sybil_str, not combined_fp, is_attack=False))

    _section("SCENARIO: Mixed traffic — N=64 Sybil + benign_001")

    sybil_64 = redistribute_queries(attacker_records, 64)
    mixed_records = sybil_64 + benign_records

    prada_mixed = run_prada_on_records(mixed_records)
    sybil_accts = [a for a in prada_mixed if a.startswith("sybil_")]
    benign_accts = [a for a in prada_mixed if not a.startswith("sybil_")]

    n_sybil_prada = sum(1 for a in sybil_accts if prada_mixed[a]["flagged"])
    n_benign_prada = sum(1 for a in benign_accts if prada_mixed[a]["flagged"])
    n_sybil_warmup = sum(1 for a in sybil_accts if prada_mixed[a]["W"] is None)

    print(f"  PRADA: Sybil {n_sybil_prada}/64 flagged ({n_sybil_warmup} warmup), "
          f"Benign {n_benign_prada}/1 flagged")

    print(f"  Sybil detect:")
    sd_mixed = run_sybil_detection(mixed_records, verbose=True)

    sybil_in_cluster = [a for a in sd_mixed["flagged_accounts"] if a.startswith("sybil_")]
    benign_in_cluster = [a for a in sd_mixed["flagged_accounts"] if not a.startswith("sybil_")]

    print(f"\n  Mixed scenario breakdown:")
    print(f"    Sybil accounts flagged by Sybil detection: {len(sybil_in_cluster)}/64")
    print(f"    Benign accounts flagged by Sybil detection: {len(benign_in_cluster)}/1"
          + ("  ← FALSE POSITIVE" if benign_in_cluster else "  ← correct (no FP)"))

    attack_caught = sd_mixed["sybil_detected"] or (n_sybil_prada > 0)
    false_positive = bool(benign_in_cluster) or bool(n_benign_prada)

    summary_rows.append(_row(
        "N=64 + benign",
        f"Sybil 0/64 ({n_sybil_warmup} warmup), Benign 0/1",
        f"Sybil {len(sybil_in_cluster)}/64, Benign {len(benign_in_cluster)}/1",
        attack_caught,
        is_attack=True,
        fp_note=("FP!" if false_positive else "no FP"),
    ))

    _js_diagnostics(attacker_records, benign_records)

    _print_summary(summary_rows)

    print("\n Evaluation complete.")
    return summary_rows


def _js_diagnostics(attacker_records: list, benign_records: list):
    _section("JS DIVERGENCE DIAGNOSTICS — threshold justification")

    sybil_64 = redistribute_queries(attacker_records, 64)
    mixed = sybil_64 + benign_records

    account_dmin = compute_dmin_per_account(mixed)
    eligible = {a: d["D"] for a, d in account_dmin.items() if len(d["D"]) >= SYBIL_MIN_DMIN}

    histograms = build_histograms(eligible)
    accounts, js_matrix = compute_pairwise_js(histograms)

    sybil_idx = [i for i, a in enumerate(accounts) if a.startswith("sybil_")]
    benign_idx = [i for i, a in enumerate(accounts) if not a.startswith("sybil_")]

    within_sybil = [js_matrix[i, j] for i in sybil_idx for j in sybil_idx if i < j]
    cross_js = [js_matrix[i, j] for i in sybil_idx for j in benign_idx]

    print(f"  Within-Sybil JS  (N=64 pairs) : "
          f"min={min(within_sybil):.4f}  "
          f"mean={np.mean(within_sybil):.4f}  "
          f"max={max(within_sybil):.4f}")

    if cross_js:
        print(f"  Sybil vs Benign JS             : "
              f"min={min(cross_js):.4f}  "
              f"mean={np.mean(cross_js):.4f}  "
              f"max={max(cross_js):.4f}")
        gap = min(cross_js) - max(within_sybil)
        print(f"  Separation gap                 : {gap:.4f}  "
              f"({'clear separation' if gap > 0 else 'OVERLAP — consider tuning threshold'})")

    t = SYBIL_JS_THRESHOLD
    threshold_ok = (
        within_sybil and cross_js
        and np.mean(within_sybil) < t < min(cross_js)
    )
    print(f"\n  SYBIL_JS_THRESHOLD = {t}  →  "
          f"mean_within={np.mean(within_sybil):.4f}  threshold={t}  min_cross={min(cross_js):.4f}")
    print(f"  Threshold valid (no false positives): "
          f"{'YES — clean separation' if threshold_ok else 'NO — OVERLAP, tighten threshold'}")


def _section(title: str):
    print(f"\n  {title}")


def _row(scenario, prada_str, sybil_str, detected, is_attack, fp_note=""):
    return {
        "scenario": scenario,
        "prada": prada_str.strip(),
        "sybil": sybil_str.strip(),
        "combined": "YES" if detected else "NO",
        "is_attack": is_attack,
        "fp_note": fp_note,
    }


def _print_summary(rows: list):
    print("\n\n  FINAL RESULTS TABLE")
    print(f"  {'Scenario':<18} {'PRADA':<30} {'Sybil Detection':<28} {'Combined'}")

    for row in rows:
        combined = row["combined"]
        if not row["is_attack"]:
            combined = "No FP" if combined == "YES" else "FP DETECTED"
        fp = f"  [{row['fp_note']}]" if row.get("fp_note") else ""
        print(f"  {row['scenario']:<18} {row['prada']:<30} {row['sybil']:<28} {combined}{fp}")

    print("\n  Key findings:")
    attack_rows = [r for r in rows if r["is_attack"]]
    benign_rows = [r for r in rows if not r["is_attack"]]

    for r in attack_rows:
        status = "DETECTED" if r["combined"] == "YES" else "MISSED"
        print(f"    {r['scenario']:<18} : {status}")

    for r in benign_rows:
        fp = r.get("fp_note", "")
        print(f"    {r['scenario']:<18} : {'NO FALSE POSITIVES' if 'FP' not in fp else 'FALSE POSITIVE!'}")

    missed = [r for r in attack_rows if r["combined"] != "YES"]
    if not missed:
        print("\n  All attack scenarios detected. Zero false positives.")
    else:
        print(f"\n  {len(missed)} scenario(s) missed detection.")


if __name__ == "__main__":
    evaluate()
