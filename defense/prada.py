# defense/prada.py
# PRADA: Protecting Against DNN Model Stealing Attacks
# Implements Algorithm 3 from paper (page 11)
# Per account, per class distance analysis + Shapiro-Wilk normality test

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import shapiro
from config import DELTA, MIN_QUERIES, LOG_PATH

# Load logs
def load_logs(log_path: Path = LOG_PATH) -> list[dict]:
    """
    Load all query logs from JSONL file.
    Each line is one query record.
    """
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

  
# Core PRADA: per account, per class
# Algorithm 3 from paper (page 11)
def compute_dmin_per_account(records: list[dict]) -> dict:
    """
    For each account, compute d_min values per class.
    
    Algorithm 3 steps:
    - Group queries by account
    - For each query, get predicted class c
    - Compute min L2 distance to previous queries of same class
    - Store in growing set Gc per class
    - Collect all d_min values in D

    Returns:
        results: dict {
            account_id: {
                "D": list of d_min values,
                "n_queries": int
            }
        }
    """
    
    # Group records by account
    by_account = defaultdict(list)
    for rec in records:
        by_account[rec["account_id"]].append(rec)

    results = {}

    for account_id, queries in by_account.items():
        # Gc: growing set per class {class: [vectors]}
        Gc = defaultdict(list)

        # Tc: threshold per class
        Tc = defaultdict(float)

        # D: all d_min values for this account
        D = []

        for query in queries:
            # Skip if no input_vector (old logs)
            if "input_vector" not in query:
                continue

            x_vec = np.array(query["input_vector"], dtype=np.float32)
            c = query["pred"]   # predicted class (Algorithm 3, line 4)

            # Algorithm 3, line 5: if Gc is empty
            if len(Gc[c]) == 0:
                Gc[c].append(x_vec)
                continue

            # Algorithm 3, lines 9-12: compute dmin
            mat = np.stack(Gc[c], axis=0)         # (n, 784)
            dists = np.linalg.norm(mat - x_vec, axis=1)
            dmin = float(np.min(dists))

            # Algorithm 3, line 13: add to D
            D.append(dmin)

            # Algorithm 3, line 14-17: update Gc and Tc
            if dmin > Tc[c]:
                Gc[c].append(x_vec)
                DGc = []
                for i in range(len(Gc[c])):
                    dists = [np.linalg.norm(Gc[c][i] - Gc[c][j])
                        for j in range(len(Gc[c])) if i != j]
                    if dists:
                        DGc.append(min(dists))

                if len(DGc) > 0:
                    Tc[c] = max(Tc[c], np.mean(DGc) - np.std(DGc))

        results[account_id] = {
            "D": D,
            "n_queries": len(queries)
        }

    return results


  
# Shapiro-Wilk normality test
# Algorithm 3, lines 20-26
def run_shapiro(D: list[float], delta: float = DELTA) -> dict:
    """
    Run Shapiro-Wilk test on d_min distribution.
    
    Algorithm 3:
    1. Remove outliers (3 std devs)
    2. Run Shapiro-Wilk
    3. If W < delta → flag as attack

    Returns:
        {
            "W": Shapiro-Wilk statistic,
            "p_value": p-value,
            "flagged": bool,
            "reason": str
        }
    """
    # Need minimum queries
    if len(D) < MIN_QUERIES:
        return {
            "W": None,
            "p_value": None,
            "flagged": False,
            "reason": f"warmup ({len(D)}/{MIN_QUERIES} queries)"
        }

    D_arr = np.array(D, dtype=np.float64)

    # Algorithm 3, line 21: remove outliers (3 std devs)
    mean = np.mean(D_arr)
    std = np.std(D_arr)
    D_clean = D_arr[np.abs(D_arr - mean) <= 3 * std]

    if len(D_clean) < 10:
        return {
            "W": None,
            "p_value": None,
            "flagged": False,
            "reason": "not enough data after outlier removal"
        }

    # Algorithm 3, line 22: Shapiro-Wilk test
    W, p_value = shapiro(D_clean)

    # Algorithm 3, line 22: if W < delta → attack
    flagged = float(W) < delta

    return {
        "W": round(float(W), 4),
        "p_value": round(float(p_value), 4),
        "flagged": flagged,
        "reason": "attack detected" if flagged else "benign"
    }


  
# Main PRADA detection
def run_prada(delta: float = DELTA, log_path: Path = LOG_PATH) -> dict:
    """
    Run PRADA detection on all accounts in log file.

    Returns:
        {
            account_id: {
                "n_queries": int,
                "n_distances": int,
                "W": float,
                "p_value": float,
                "flagged": bool,
                "reason": str
            }
        }
    """
    print("=" * 50)
    print("PRADA Detection (Algorithm 3)")
    print(f"δ threshold: {delta}")
    print("=" * 50)

    # Load logs
    records = load_logs(log_path)
    print(f"\nLoaded {len(records)} query records")

    # Compute d_min per account
    account_results = compute_dmin_per_account(records)

    # Run Shapiro-Wilk per account
    final_results = {}

    print(f"\n{'Account':<20} {'Queries':<10} {'Distances':<12} {'W':<8} {'Flagged'}")
    print("-" * 60)

    for account_id, data in account_results.items():
        D = data["D"]
        n_queries = data["n_queries"]

        shapiro_result = run_shapiro(D, delta)

        final_results[account_id] = {
            "n_queries": n_queries,
            "n_distances": len(D),
            "W": shapiro_result["W"],
            "p_value": shapiro_result["p_value"],
            "flagged": shapiro_result["flagged"],
            "reason": shapiro_result["reason"]
        }

        W_str = str(shapiro_result["W"]) if shapiro_result["W"] else "N/A"
        flagged_str = "🚨 ATTACK" if shapiro_result["flagged"] else "✅ benign"

        print(f"{account_id:<20} {n_queries:<10} {len(D):<12} {W_str:<8} {flagged_str}")

    return final_results


  
# Entry point
if __name__ == "__main__":
    results = run_prada()
    print("\n[Done] PRADA detection complete")