"""
analysis/generate_results.py — Compute and save all missing JSON result files.

Skips any file that already exists. No server needed.
"""

import json
import sys
import warnings
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from defense.logs import load_logs
from defense.prada import run_prada_on_records
from defense.sybil_detection import run_sybil_detection
from simulation.sybil import redistribute_queries
from config import (
    LOG_PATH, DELTA,
    SYBIL_JS_THRESHOLD, SYBIL_MIN_CLUSTER, SYBIL_MIN_DMIN,
)

RESULTS_DIR    = ROOT / "analysis" / "results"
ATTACKER_LIMIT = 6400
BENIGN_LIMIT   = 3000
N_SWEEP        = [4, 8, 16, 32, 56, 64, 128, 192, 256]
THRESH_SWEEP   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_FIXED        = 64


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Saved → {path.relative_to(ROOT)}")


def js_stats(result: dict) -> tuple:
    """(within_js, cross_js) from a run_sybil_detection result."""
    if result["js_matrix"] is None:
        return None, None
    accounts = result["accounts"]
    mat      = result["js_matrix"]
    sybil_idx  = [i for i, a in enumerate(accounts) if     a.startswith("sybil_")]
    benign_idx = [i for i, a in enumerate(accounts) if not a.startswith("sybil_")]
    within = [mat[i, j] for i in sybil_idx for j in sybil_idx if i < j]
    cross  = [mat[i, j] for i in sybil_idx for j in benign_idx]
    within_js = float(np.mean(within)) if within else None
    cross_js  = float(np.mean(cross))  if cross  else None
    return within_js, cross_js


def sybil_detect(records):
    return run_sybil_detection(
        records,
        js_threshold=SYBIL_JS_THRESHOLD,
        min_cluster=SYBIL_MIN_CLUSTER,
        min_dmin=SYBIL_MIN_DMIN,
        verbose=False,
    )


def load_data():
    all_records = load_logs(LOG_PATH)
    attacker = [r for r in all_records if r["account_id"] == "attacker_001"][:ATTACKER_LIMIT]
    cvsearch = [r for r in all_records if r["account_id"] == "attacker_cvsearch"][:ATTACKER_LIMIT]
    mixed    = [r for r in all_records if r["account_id"] == "mixed_sybil_source"][:ATTACKER_LIMIT]
    benign   = [r for r in all_records if r["account_id"] == "benign_001"][:BENIGN_LIMIT]
    return attacker, cvsearch, mixed, benign


# ── 1. prada_baseline.json ────────────────────────────────────────────────────

def gen_prada_baseline(attacker, cvsearch, benign, out_path):
    if out_path.exists():
        print(f"  Skip (exists): {out_path.name}")
        return
    rows = []
    for records in [attacker, cvsearch, benign]:
        if not records:
            continue
        prada = run_prada_on_records(records, DELTA)
        for acct, r in prada.items():
            rows.append({"account_id": acct, "W": r["W"], "flagged": bool(r["flagged"])})
    save(out_path, rows)


# ── 2. prada_n_sweep.json ─────────────────────────────────────────────────────

def gen_prada_n_sweep(attacker, out_path):
    if out_path.exists():
        print(f"  Skip (exists): {out_path.name}")
        return
    rows = []
    for N in N_SWEEP:
        recs          = redistribute_queries(attacker, N)
        prada         = run_prada_on_records(recs, DELTA)
        flagged_count = sum(1 for r in prada.values() if r["flagged"])
        warmup_count  = sum(1 for r in prada.values() if r["W"] is None)
        rows.append({
            "N": N, "qpa": ATTACKER_LIMIT // N,
            "flagged_count": flagged_count,
            "warmup_count":  warmup_count,
            "detection_pct": flagged_count / N * 100,
        })
    save(out_path, rows)


# ── 3. js_n_sweep.json ────────────────────────────────────────────────────────

def gen_js_n_sweep(attacker, benign, out_path):
    if out_path.exists():
        print(f"  Skip (exists): {out_path.name}")
        return
    rows = []
    for N in N_SWEEP:
        recs     = redistribute_queries(attacker, N)
        result   = sybil_detect(recs + benign)
        wjs, cjs = js_stats(result)
        gap      = (cjs - wjs) if (wjs is not None and cjs is not None) else None
        rows.append({
            "N": N, "qpa": ATTACKER_LIMIT // N,
            "eligible":  result["n_eligible"],
            "within_js": wjs, "cross_js": cjs, "gap": gap,
            "detected":  result["sybil_detected"],
            "FP":        "benign_001" in result["flagged_accounts"],
        })
    save(out_path, rows)


# ── 4. combined_n_sweep.json ──────────────────────────────────────────────────

def gen_combined_n_sweep(attacker, benign, out_path):
    if out_path.exists():
        print(f"  Skip (exists): {out_path.name}")
        return
    prada_benign_fp = run_prada_on_records(benign, DELTA).get("benign_001", {}).get("flagged", False)
    rows = []
    for N in N_SWEEP:
        recs           = redistribute_queries(attacker, N)
        prada_detected = any(r["flagged"] for r in run_prada_on_records(recs, DELTA).values())
        js_result      = sybil_detect(recs + benign)
        js_detected    = js_result["sybil_detected"]
        js_fp          = "benign_001" in js_result["flagged_accounts"]
        rows.append({
            "N": N,
            "prada_detected": bool(prada_detected),
            "js_detected":    bool(js_detected),
            "combined":       bool(prada_detected or js_detected),
            "FP":             bool(prada_benign_fp or js_fp),
        })
    save(out_path, rows)


# ── 5. mixed_sybil_n_sweep.json ───────────────────────────────────────────────

def gen_mixed_sybil_n_sweep(mixed, benign, out_path):
    if out_path.exists():
        print(f"  Skip (exists): {out_path.name}")
        return
    prada_benign_fp = run_prada_on_records(benign, DELTA).get("benign_001", {}).get("flagged", False)
    rows = []
    for N in N_SWEEP:
        recs          = redistribute_queries(mixed, N)
        prada         = run_prada_on_records(recs, DELTA)
        flagged_count = sum(1 for r in prada.values() if r["flagged"])
        js_result     = sybil_detect(recs + benign)
        js_detected   = js_result["sybil_detected"]
        js_fp         = "benign_001" in js_result["flagged_accounts"]
        rows.append({
            "N": N, "qpa": ATTACKER_LIMIT // N,
            "prada_flagged_count": flagged_count,
            "js_detected":         bool(js_detected),
            "combined":            bool(flagged_count > 0 or js_detected),
            "FP":                  bool(prada_benign_fp or js_fp),
        })
    save(out_path, rows)


# ── 6. js_threshold_sweep.json ────────────────────────────────────────────────

def gen_js_threshold_sweep(attacker, mixed, benign, out_path):
    if out_path.exists():
        print(f"  Skip (exists): {out_path.name}")
        return
    rows = []
    sources = [("pure", attacker), ("mixed", mixed)]
    for source_name, source_records in sources:
        if not source_records:
            continue
        combined = redistribute_queries(source_records, N_FIXED) + benign
        for thresh in THRESH_SWEEP:
            result   = run_sybil_detection(
                combined,
                js_threshold=thresh,
                min_cluster=SYBIL_MIN_CLUSTER,
                min_dmin=SYBIL_MIN_DMIN,
                verbose=False,
            )
            wjs, cjs = js_stats(result)
            gap      = (cjs - wjs) if (wjs is not None and cjs is not None) else None
            rows.append({
                "threshold": thresh, "source": source_name,
                "within_js": wjs, "cross_js": cjs, "gap": gap,
                "detected":  bool(result["sybil_detected"]),
                "FP":        bool("benign_001" in result["flagged_accounts"]),
            })
    save(out_path, rows)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading records...")
    attacker, cvsearch, mixed, benign = load_data()
    print(f"  attacker_001:       {len(attacker)} records")
    print(f"  attacker_cvsearch:  {len(cvsearch)} records")
    print(f"  mixed_sybil_source: {len(mixed)} records")
    print(f"  benign_001:         {len(benign)} records")
    print()

    tasks = [
        ("prada_baseline.json",
         lambda: gen_prada_baseline(attacker, cvsearch, benign,
                                    RESULTS_DIR / "prada_baseline.json")),
        ("prada_n_sweep.json",
         lambda: gen_prada_n_sweep(attacker,
                                   RESULTS_DIR / "prada_n_sweep.json")),
        ("js_n_sweep.json",
         lambda: gen_js_n_sweep(attacker, benign,
                                RESULTS_DIR / "js_n_sweep.json")),
        ("combined_n_sweep.json",
         lambda: gen_combined_n_sweep(attacker, benign,
                                      RESULTS_DIR / "combined_n_sweep.json")),
        ("mixed_sybil_n_sweep.json",
         lambda: gen_mixed_sybil_n_sweep(mixed, benign,
                                         RESULTS_DIR / "mixed_sybil_n_sweep.json")),
        ("js_threshold_sweep.json",
         lambda: gen_js_threshold_sweep(attacker, mixed, benign,
                                        RESULTS_DIR / "js_threshold_sweep.json")),
    ]

    for name, fn in tasks:
        print(f"[{name}]")
        fn()
        print()

    print("[Done] generate_results.py complete.")
