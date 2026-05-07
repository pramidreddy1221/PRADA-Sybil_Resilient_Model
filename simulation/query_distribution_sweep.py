import sys
import json
import random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defense.logs import load_logs
from defense.sybil_detection import run_sybil_detection
from config import MIN_QUERIES, SYBIL_JS_THRESHOLD, SYBIL_MIN_DMIN, SYBIL_MIN_CLUSTER

N_SYBIL = 64
ATTACKER_LIMIT = 6400
BENIGN_LIMIT = 3000
N_BENIGN = BENIGN_LIMIT // MIN_QUERIES


def redistribute_rr(records, n_accounts, prefix):
    ids = [f"{prefix}{i:03d}" for i in range(1, n_accounts + 1)]
    result = []
    for i, rec in enumerate(records):
        new_rec = dict(rec)
        new_rec["account_id"] = ids[i % n_accounts]
        result.append(new_rec)
    return result


def build_mixed(attacker_records, benign_records, n_accounts, prefix):
    n_atk_per = len(attacker_records) // n_accounts
    n_ben_per = round(n_atk_per * 7 / 3)  # 70-30 mix: 7 benign queries for every 3 attacker queries per account
    result = []
    for i in range(n_accounts):
        acct_id = f"{prefix}{i + 1:03d}"
        atk = attacker_records[i * n_atk_per : (i + 1) * n_atk_per]
        ben = [benign_records[(i * n_ben_per + j) % len(benign_records)] for j in range(n_ben_per)]
        merged = []
        ai, bi = 0, 0
        na, nb = len(atk), len(ben)
        # Interleave attacker and benign queries 3:7 per block to simulate mixed-strategy Sybil account
        while ai < na or bi < nb:
            for _ in range(3):
                if ai < na:
                    rec = dict(atk[ai])
                    rec["account_id"] = acct_id
                    merged.append(rec)
                    ai += 1
            for _ in range(7):
                if bi < nb:
                    rec = dict(ben[bi])
                    rec["account_id"] = acct_id
                    merged.append(rec)
                    bi += 1
        result.extend(merged)
    return result


def run_distribution(name, sybil_recs, benign_recs):
    result = run_sybil_detection(sybil_recs + benign_recs, verbose=False)
    within_js = result["mean_js_within"]
    cross_js = result["mean_js_cross"]
    gap = (cross_js - within_js) if (within_js is not None and cross_js is not None) else None
    accounts = result["accounts"]
    elig_s = sum(1 for a in accounts if not a.startswith("benign_"))
    elig_b = sum(1 for a in accounts if a.startswith("benign_"))
    fp = sum(1 for a in result["flagged_accounts"] if a.startswith("benign_"))
    return {
        "name": name,
        "elig_s": elig_s,
        "elig_b": elig_b,
        "within_js": within_js,
        "cross_js": cross_js,
        "gap": gap,
        "detected": result["sybil_detected"],
        "fp": fp,
    }


def fj(v):
    return f"{v:.4f}" if v is not None else "   N/A"


def main():
    all_records = load_logs()
    attacker_records = [r for r in all_records if r["account_id"] == "attacker_001"][:ATTACKER_LIMIT]
    benign_records = [r for r in all_records if r["account_id"] == "benign_001"][:BENIGN_LIMIT]

    print(f"attacker_001: {len(attacker_records)} records")
    print(f"benign_001: {len(benign_records)} records")
    print(f"N_SYBIL={N_SYBIL}, N_BENIGN={N_BENIGN}, JS_THRESHOLD={SYBIL_JS_THRESHOLD}")

    benign_recs = redistribute_rr(benign_records, N_BENIGN, "benign_")

    rr_sybil = redistribute_rr(attacker_records, N_SYBIL, "sybil_rr_")

    shuffled = list(attacker_records)
    random.seed(42)
    random.shuffle(shuffled)
    rand_sybil = redistribute_rr(shuffled, N_SYBIL, "sybil_rand_")

    mix_sybil = build_mixed(attacker_records, benign_records, N_SYBIL, "sybil_mix_")

    runs = [
        run_distribution("round-robin", rr_sybil, benign_recs),
        run_distribution("randomized", rand_sybil, benign_recs),
        run_distribution("mixed-70-30", mix_sybil, benign_recs),
    ]

    hdr = (
        f"{'Distribution':<14}  {'Elig-S':>6}  {'Elig-B':>6}"
        f"  {'JS-Within':>10}  {'JS-Cross':>10}  {'Gap':>8}"
        f"  {'Detected':>8}  {'FP':>3}"
    )
    print(hdr)
    for r in runs:
        print(
            f"{r['name']:<14}  {r['elig_s']:>6}  {r['elig_b']:>6}"
            f"  {fj(r['within_js']):>10}  {fj(r['cross_js']):>10}  {fj(r['gap']):>8}"
            f"  {'yes' if r['detected'] else 'no':>8}  {r['fp']:>3}"
        )

    out_path = ROOT / "simulation" / "query_distribution_results.json"
    out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
