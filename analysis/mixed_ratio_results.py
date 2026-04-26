"""
analysis/mixed_ratio_results.py — PRADA W scores for mixed-ratio accounts.

Loads mixed_010..mixed_090 from the query log (first 6400 records each),
runs PRADA, and saves results to analysis/results/mixed_ratio.json.
No server needed.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from defense.logs import load_logs
from defense.prada import run_prada_on_records
from config import LOG_PATH

ACCOUNTS = ["mixed_010", "mixed_020", "mixed_030", "mixed_050", "mixed_070", "mixed_090"]
RATIOS = {
    "mixed_010": 0.10, "mixed_020": 0.20, "mixed_030": 0.30,
    "mixed_050": 0.50, "mixed_070": 0.70, "mixed_090": 0.90,
}
LIMIT = 6400


if __name__ == "__main__":
    print(f"Loading logs from {LOG_PATH}")
    all_records = load_logs(LOG_PATH)

    per_account: dict = {a: [] for a in ACCOUNTS}
    for r in all_records:
        aid = r["account_id"]
        if aid in per_account and len(per_account[aid]) < LIMIT:
            per_account[aid].append(r)

    records_to_eval = [r for rows in per_account.values() for r in rows]
    n_accounts = sum(1 for v in per_account.values() if v)
    print(f"  Loaded {len(records_to_eval)} records across {n_accounts} accounts")

    prada = run_prada_on_records(records_to_eval)

    rows = []
    for account_id in ACCOUNTS:
        res = prada.get(account_id, {})
        w = res.get("W")
        flagged = res.get("flagged", False)
        ratio = RATIOS[account_id]
        rows.append({
            "account_id": account_id,
            "normal_ratio": ratio,
            "W": w,
            "flagged": flagged,
        })
        w_str = f"{w:.4f}" if w is not None else "warmup"
        flag_str = "YES" if flagged else "NO"
        print(f"  {account_id:<14}  ratio={ratio:.2f}  W={w_str}  flagged={flag_str}")

    out_path = ROOT / "analysis" / "results" / "mixed_ratio.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path.relative_to(ROOT)}")
