# Baseline comparison — merges all Sybil streams into one account and runs PRADA
# on the combined sequence. Requires knowing which accounts belong to the attacker,
# which is not available in real deployment.
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defense.logs import load_logs
from defense.prada import run_prada_on_records
from config import LOG_PATH, DELTA, MIN_QUERIES

N_VALUES = [4, 8, 16, 32, 64, 128, 256]
ATTACKER_LIMIT = 6400


def redistribute(records, n_accounts, prefix):
    result = []
    for i, rec in enumerate(records):
        new_rec = dict(rec)
        new_rec["account_id"] = f"{prefix}{(i % n_accounts) + 1:03d}"
        result.append(new_rec)
    return result


def main():
    all_records = load_logs()
    attacker_pool = [r for r in all_records if r["account_id"] == "attacker_001"][:ATTACKER_LIMIT]
    print(f"attacker_001: {len(attacker_pool)} records")

    js_path = ROOT / "analysis" / "results" / "js_n_sweep.json"
    js_by_n = {}
    if js_path.exists():
        js_data = json.loads(js_path.read_text(encoding="utf-8"))
        js_by_n = {entry["N"]: entry["detected"] for entry in js_data}

    rows = []

    for N in N_VALUES:
        sybil_recs = redistribute(attacker_pool, N, "sybil_")
        qpa = len(attacker_pool) // N
        total_combined = len(sybil_recs)

        combined_recs = []
        for rec in sybil_recs:
            new_rec = dict(rec)
            new_rec["account_id"] = "naive_combined"
            combined_recs.append(new_rec)

        combined_result = run_prada_on_records(combined_recs, DELTA)
        combined_entry = combined_result.get("naive_combined", {})
        combined_w = combined_entry.get("W")
        naive_flagged = combined_entry.get("flagged", False)

        per_account_result = run_prada_on_records(sybil_recs, DELTA)
        n_flagged = sum(1 for v in per_account_result.values() if v["flagged"])

        js_detected = js_by_n.get(N)

        rows.append({
            "N": N,
            "qpa": qpa,
            "total_combined_queries": total_combined,
            "combined_W": combined_w,
            "naive_flagged": naive_flagged,
            "per_account_flagged": n_flagged,
            "per_account_total": N,
            "js_detected": js_detected,
        })

    print(f"\n{'N':<6} {'QPA':<6} {'Combined W':<12} {'Naive Flagged':<15} {'Per-Acct Flagged':<18} {'JS Detected'}")
    for row in rows:
        w_str = f"{row['combined_W']:.4f}" if row["combined_W"] is not None else "N/A"
        js_str = str(row["js_detected"]) if row["js_detected"] is not None else "N/A"
        per_str = f"{row['per_account_flagged']}/{row['per_account_total']}"
        print(f"{row['N']:<6} {row['qpa']:<6} {w_str:<12} {str(row['naive_flagged']):<15} {per_str:<18} {js_str}")

    out_path = ROOT / "simulation" / "naive_prada_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
