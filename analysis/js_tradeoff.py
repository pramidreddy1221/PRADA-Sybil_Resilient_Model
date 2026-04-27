import json
import sys
import warnings
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from defense.prada import run_prada_on_records
from defense.sybil_detection import run_sybil_detection
from simulation.sybil import redistribute_queries
from config import (
    DELTA, MIN_QUERIES,
    SYBIL_JS_THRESHOLD, SYBIL_MIN_CLUSTER, SYBIL_MIN_DMIN, SYBIL_N_BINS,
)

LOG_PATH = ROOT / "logs" / "queries.jsonl"
N_SWEEP = [4, 8, 16, 32, 56, 64, 128, 160, 192, 256]
THRESH_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_TOTAL = 6400
BENIGN_LIMIT = 3000
N_FIXED = 64


def load_data() -> tuple:
    attacker, mixed, benign = [], [], []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = rec["account_id"]
            if   aid == "attacker_001"       and len(attacker) < N_TOTAL:
                attacker.append(rec)
            elif aid == "mixed_sybil_source" and len(mixed)    < N_TOTAL:
                mixed.append(rec)
            elif aid == "benign_001"         and len(benign)   < BENIGN_LIMIT:
                benign.append(rec)
    return attacker, mixed, benign


def compute_js_stats(result: dict) -> tuple:
    if result["js_matrix"] is None:
        return None, None

    accounts = result["accounts"]
    mat = result["js_matrix"]

    sybil_idx = [i for i, a in enumerate(accounts) if     a.startswith("sybil_")]
    benign_idx = [i for i, a in enumerate(accounts) if not a.startswith("sybil_")]

    within_pairs = [mat[i, j] for i in sybil_idx for j in sybil_idx if i < j]
    cross_pairs = [mat[i, j] for i in sybil_idx for j in benign_idx]

    within_js = float(np.mean(within_pairs)) if within_pairs else None
    cross_js = float(np.mean(cross_pairs))  if cross_pairs  else None
    return within_js, cross_js


def yn(flag: bool) -> str:
    return "YES" if flag else "no"

def fj(v) -> str:
    return f"{v:.4f}" if v is not None else "  N/A"

def div(w: int) -> str:
    return "─" * w


def table1(attacker: list) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TABLE 1: PRADA alone — N sweep (pure Sybil)                    ║")
    print("║  Fixed: DELTA=0.96, MIN_QUERIES=100                             ║")
    print("║  Source: attacker_001 first 6400 records                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    hdr = f"{'N':>6}  {'qpa':>6}  {'Flagged/N':>10}  {'Warmup':>8}  {'Detect%':>8}"
    w = len(hdr)
    print(hdr)

    rows = []
    for N in N_SWEEP:
        qpa = N_TOTAL // N
        recs = redistribute_queries(attacker, N)
        prada = run_prada_on_records(recs, DELTA)
        n_flagged = sum(1 for r in prada.values() if r["flagged"])
        n_warmup = sum(1 for r in prada.values() if r["W"] is None)
        detect_pct = n_flagged / N * 100
        print(f"{N:>6}  {qpa:>6}  {n_flagged:>4}/{N:<5}  {n_warmup:>8}  {detect_pct:>7.1f}%")
        rows.append({"N": N, "qpa": qpa, "n_flagged": n_flagged, "n_warmup": n_warmup, "detect_pct": detect_pct})

    print("  qpa = queries per account (6400 / N)")
    print("  Warmup = accounts where len(D) < MIN_QUERIES → Shapiro-Wilk never runs")
    print("  Gap: at N=64, qpa=100 but ~90 dmin distances → all accounts in warmup")
    return rows


def table2(attacker: list, benign: list) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TABLE 2: JS alone — N sweep (pure Sybil)                       ║")
    print("║  Fixed: JS threshold=0.15, min_cluster=3, min_dmin=10           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    hdr = (f"{'N':>6}  {'qpa':>6}  {'Eligible':>8}  "
           f"{'JS_within':>10}  {'JS_sybben':>10}  {'Gap':>8}  "
           f"{'Det':>5}  {'FP':>5}")
    w = len(hdr)
    print(hdr)

    rows = []
    for N in N_SWEEP:
        qpa = N_TOTAL // N
        recs = redistribute_queries(attacker, N)
        combined = recs + benign
        result = run_sybil_detection(
            combined,
            js_threshold=SYBIL_JS_THRESHOLD,
            min_cluster=SYBIL_MIN_CLUSTER,
            min_dmin=SYBIL_MIN_DMIN,
            verbose=False,
        )
        within_js, cross_js = compute_js_stats(result)
        gap = (cross_js - within_js) if (within_js is not None and cross_js is not None) else None
        detected = yn(result["sybil_detected"])
        fp = yn("benign_001" in result["flagged_accounts"])

        print(
            f"{N:>6}  {qpa:>6}  {result['n_eligible']:>8}  "
            f"{fj(within_js):>10}  {fj(cross_js):>10}  {fj(gap):>8}  "
            f"{detected:>5}  {fp:>5}"
        )
        rows.append({
            "N": N, "qpa": qpa, "n_eligible": result["n_eligible"],
            "within_js": within_js, "cross_js": cross_js, "gap": gap,
            "detected": result["sybil_detected"],
            "fp": "benign_001" in result["flagged_accounts"],
        })

    print("  JS_within = mean JS between all pairs of Sybil accounts (lower = more coordinated)")
    print("  JS_sybben = mean JS between Sybil accounts and benign_001 (higher = clearer separation)")
    print("  Gap = JS_sybben − JS_within  (larger = easier to detect)")
    print("  FP = benign_001 erroneously included in flagged Sybil cluster")
    return rows


def table3(attacker: list, benign: list, prada_benign_fp: bool) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TABLE 3: Combined PRADA + JS — N sweep (pure Sybil)            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    hdr = f"{'N':>6}  {'PRADA_det':>10}  {'JS_det':>7}  {'Combined':>9}  {'FP':>5}"
    w = len(hdr)
    print(hdr)

    rows = []
    for N in N_SWEEP:
        recs = redistribute_queries(attacker, N)

        prada_res = run_prada_on_records(recs, DELTA)
        prada_detected = any(r["flagged"] for r in prada_res.values())

        combined = recs + benign
        js_result = run_sybil_detection(
            combined,
            js_threshold=SYBIL_JS_THRESHOLD,
            min_cluster=SYBIL_MIN_CLUSTER,
            min_dmin=SYBIL_MIN_DMIN,
            verbose=False,
        )
        js_detected = js_result["sybil_detected"]
        js_fp = "benign_001" in js_result["flagged_accounts"]

        combined_det = prada_detected or js_detected
        fp = prada_benign_fp or js_fp

        print(
            f"{N:>6}  {yn(prada_detected):>10}  {yn(js_detected):>7}  "
            f"{yn(combined_det):>9}  {yn(fp):>5}"
        )
        rows.append({
            "N": N, "prada_detected": prada_detected,
            "js_detected": js_detected, "combined_detected": combined_det, "fp": fp,
        })

    print("  Combined = PRADA OR JS detected")
    print("  FP = benign_001 flagged by PRADA (W<δ) OR included in JS Sybil cluster")
    print(f"  PRADA benign_001 FP (constant across N): {yn(prada_benign_fp)}")
    return rows


def table4(mixed: list, benign: list, prada_benign_fp: bool) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TABLE 4: Mixed Sybil — N sweep (combined PRADA + JS)           ║")
    print("║  Source: mixed_sybil_source (30% normal queries per round)      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    hdr = (f"{'N':>6}  {'qpa':>6}  {'PRADA_fl/N':>12}  "
           f"{'JS_det':>7}  {'Combined':>9}  {'FP':>5}")
    w = len(hdr)
    print(hdr)

    rows = []
    for N in N_SWEEP:
        qpa = N_TOTAL // N
        recs = redistribute_queries(mixed, N)

        prada_res = run_prada_on_records(recs, DELTA)
        n_flagged = sum(1 for r in prada_res.values() if r["flagged"])

        combined = recs + benign
        js_result = run_sybil_detection(
            combined,
            js_threshold=SYBIL_JS_THRESHOLD,
            min_cluster=SYBIL_MIN_CLUSTER,
            min_dmin=SYBIL_MIN_DMIN,
            verbose=False,
        )
        js_detected = js_result["sybil_detected"]
        js_fp = "benign_001" in js_result["flagged_accounts"]

        combined_det = (n_flagged > 0) or js_detected
        fp = prada_benign_fp or js_fp

        print(
            f"{N:>6}  {qpa:>6}  {n_flagged:>4}/{N:<7}  "
            f"{yn(js_detected):>7}  {yn(combined_det):>9}  {yn(fp):>5}"
        )
        rows.append({
            "N": N, "qpa": qpa, "prada_flagged": n_flagged,
            "js_detected": js_detected, "combined_detected": combined_det, "fp": fp,
        })

    print("  mixed_sybil_source: attacker injects 30% normal images per round to dilute")
    print("  the dmin pattern — compare detection rates here vs Table 3 (pure Sybil)")
    print(f"  PRADA benign_001 FP (constant across N): {yn(prada_benign_fp)}")
    return rows


def table5(attacker: list, benign: list) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  TABLE 5: JS threshold sweep — N={N_FIXED} pure Sybil               ║")
    print("║  Source: attacker_001 | Fixed N=64, min_cluster=3              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    recs = redistribute_queries(attacker, N_FIXED)
    combined = recs + benign

    hdr = (f"{'Threshold':>10}  {'JS_within':>10}  {'JS_sybben':>10}  "
           f"{'Gap':>8}  {'Det':>5}  {'FP':>5}")
    w = len(hdr)
    print(hdr)

    rows = []
    for thresh in THRESH_SWEEP:
        result = run_sybil_detection(
            combined,
            js_threshold=thresh,
            min_cluster=SYBIL_MIN_CLUSTER,
            min_dmin=SYBIL_MIN_DMIN,
            verbose=False,
        )
        within_js, cross_js = compute_js_stats(result)
        gap = (cross_js - within_js) if (within_js is not None and cross_js is not None) else None

        print(
            f"{thresh:>10.2f}  {fj(within_js):>10}  {fj(cross_js):>10}  "
            f"{fj(gap):>8}  {yn(result['sybil_detected']):>5}  "
            f"{yn('benign_001' in result['flagged_accounts']):>5}"
        )
        rows.append({
            "threshold": thresh, "within_js": within_js, "cross_js": cross_js, "gap": gap,
            "detected": result["sybil_detected"],
            "fp": "benign_001" in result["flagged_accounts"],
        })

    print("  JS_within / JS_sybben are constant across threshold rows (data doesn't change)")
    print("  only Det and FP change as the threshold tightens or relaxes")
    print("  Low threshold = strict (harder to fire) | High threshold = lenient (FP risk)")
    return rows


def table6(mixed: list, benign: list) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  TABLE 6: JS threshold sweep — N={N_FIXED} mixed Sybil              ║")
    print("║  Source: mixed_sybil_source | Fixed N=64, min_cluster=3        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    recs = redistribute_queries(mixed, N_FIXED)
    combined = recs + benign

    hdr = (f"{'Threshold':>10}  {'JS_within':>10}  {'JS_sybben':>10}  "
           f"{'Gap':>8}  {'Det':>5}  {'FP':>5}")
    w = len(hdr)
    print(hdr)

    rows = []
    for thresh in THRESH_SWEEP:
        result = run_sybil_detection(
            combined,
            js_threshold=thresh,
            min_cluster=SYBIL_MIN_CLUSTER,
            min_dmin=SYBIL_MIN_DMIN,
            verbose=False,
        )
        within_js, cross_js = compute_js_stats(result)
        gap = (cross_js - within_js) if (within_js is not None and cross_js is not None) else None

        print(
            f"{thresh:>10.2f}  {fj(within_js):>10}  {fj(cross_js):>10}  "
            f"{fj(gap):>8}  {yn(result['sybil_detected']):>5}  "
            f"{yn('benign_001' in result['flagged_accounts']):>5}"
        )
        rows.append({
            "threshold": thresh, "within_js": within_js, "cross_js": cross_js, "gap": gap,
            "detected": result["sybil_detected"],
            "fp": "benign_001" in result["flagged_accounts"],
        })

    print("  Compare JS_within here vs Table 5: mixed Sybil's 30% normal injection")
    print("  raises JS_within (less coordinated), reducing the gap and making detection harder")
    return rows


if __name__ == "__main__":
    print("Loading records...")
    attacker, mixed, benign = load_data()
    print(f"  attacker_001:        {len(attacker)} records")
    print(f"  mixed_sybil_source:  {len(mixed)} records")
    print(f"  benign_001:          {len(benign)} records")

    prada_benign = run_prada_on_records(benign, DELTA)
    prada_benign_fp = prada_benign.get("benign_001", {}).get("flagged", False)
    print(f"  PRADA benign_001 baseline: W={prada_benign.get('benign_001', {}).get('W')}, "
          f"flagged={yn(prada_benign_fp)}")

    t1 = table1(attacker)
    t2 = table2(attacker, benign)
    t3 = table3(attacker, benign, prada_benign_fp)
    t4 = table4(mixed,    benign, prada_benign_fp)
    t5 = table5(attacker, benign)
    t6 = table6(mixed,    benign)

    out = {
        "table1_prada_n_sweep_pure": t1,
        "table2_js_n_sweep_pure": t2,
        "table3_combined_n_sweep_pure": t3,
        "table4_combined_n_sweep_mixed": t4,
        "table5_js_thresh_pure": t5,
        "table6_js_thresh_mixed": t6,
    }
    out_path = ROOT / "analysis" / "results" / "js_tradeoff.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path.relative_to(ROOT)}")

    print()
    print("[Done] js_tradeoff.py complete.")
