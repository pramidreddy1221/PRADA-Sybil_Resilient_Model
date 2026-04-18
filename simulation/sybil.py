"""
simulation/sybil.py — Sybil Attack Simulation

Demonstrates PRADA's per-account blindspot.

The Problem (in plain English):
  PRADA is like a bank security guard who watches ONE customer at a time.
  If a thief splits their suspicious behaviour across 10 fake customer
  accounts, each individual account looks innocent.  This script proves
  that PRADA misses the attack when queries are spread across N accounts.

Algorithm:
  1. Load the 6 400 real attacker_001 queries from the existing log
     (they include image vectors + victim labels — no re-attack needed).
  2. For each N in [2, 5, 10]:
       a. Reassign account_ids round-robin:
          query 0 → sybil_001, query 1 → sybil_002, ..., query N → sybil_001, ...
       b. Run PRADA's dmin calculation on the reassigned records (in-memory,
          no server required).
       c. Run Shapiro-Wilk test per account (δ = 0.95).
       d. Record: accounts flagged / missed, W score per account.
  3. Print the comparison table.
  4. Identify the split level at which PRADA completely fails (0 flagged).

Round-robin is the most evasion-optimal Sybil strategy: each account
receives queries spread across all attack rounds, so no single account
accumulates the concentrated FGSM pattern that PRADA looks for.
"""

import sys
import json
from pathlib import Path

# --- make sure the project root is on sys.path when run as a script ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defense.logs import load_logs
from defense.distances import compute_dmin_per_account
from defense.detection import run_shapiro
from config import LOG_PATH, DELTA, MIN_QUERIES

SOURCE_ACCOUNT = "attacker_001"
# N=2, 5, 10 — the main thesis splits.
# N=64 — warmup failure: 6400 / 64 = 100 queries/acct, but dmin list has
#         ~90 distances (first query per class skipped) → < MIN_QUERIES → warmup.
SYBIL_SPLITS   = [2, 5, 10, 64]


# ---------------------------------------------------------------------------
# Core simulation logic
# ---------------------------------------------------------------------------

def redistribute_queries(records: list[dict], n_accounts: int) -> list[dict]:
    """
    Take a flat list of query records and reassign account_ids round-robin
    across n_accounts sybil identities.

    Example (n=3):
      record 0  → sybil_001
      record 1  → sybil_002
      record 2  → sybil_003
      record 3  → sybil_001   (wraps around)
      ...
    """
    sybil_ids = [f"sybil_{i:03d}" for i in range(1, n_accounts + 1)]
    result = []
    for i, rec in enumerate(records):
        new_rec = dict(rec)                    # shallow copy — don't mutate original
        new_rec["account_id"] = sybil_ids[i % n_accounts]
        result.append(new_rec)
    return result


def run_prada_on_records(records: list[dict], delta: float = DELTA) -> dict:
    """
    Run the full PRADA detection pipeline on an in-memory list of records.
    Returns per-account results keyed by account_id.
    """
    account_dmin = compute_dmin_per_account(records)
    results = {}
    for account_id, data in account_dmin.items():
        shapiro = run_shapiro(data["D"], delta)
        results[account_id] = {
            "n_queries":    data["n_queries"],
            "n_distances":  len(data["D"]),
            "W":            shapiro["W"],
            "p_value":      shapiro["p_value"],
            "flagged":      shapiro["flagged"],
            "reason":       shapiro["reason"],
        }
    return results


def run_sybil_experiment(
    n_accounts_list: list[int] = SYBIL_SPLITS,
    source_account:  str       = SOURCE_ACCOUNT,
    log_path:        Path      = LOG_PATH,
    delta:           float     = DELTA,
) -> dict:
    """
    Run the full Sybil failure demonstration.

    Returns a dict keyed by N (number of sybil accounts) where each value
    contains aggregate + per-account PRADA results.
    """
    # -----------------------------------------------------------------------
    # 1. Load source queries
    # -----------------------------------------------------------------------
    all_records      = load_logs(log_path)
    attacker_records = [r for r in all_records
                        if r.get("account_id") == source_account]

    print("=" * 60)
    print("Sybil Attack Simulation — PRADA Failure Demonstration")
    print(f"δ threshold : {delta}")
    print(f"MIN_QUERIES : {MIN_QUERIES}")
    print(f"Source      : {source_account}  ({len(attacker_records)} queries)")
    print(f"Splits      : {n_accounts_list}")
    print("=" * 60)

    if not attacker_records:
        print(f"\n[ERROR] No records found for account '{source_account}'.")
        print("  Run the Papernot attack first (python -m attacker.attack).")
        sys.exit(1)

    experiment_results = {}

    # -----------------------------------------------------------------------
    # 2. For each N: redistribute → PRADA → record
    # -----------------------------------------------------------------------
    for N in n_accounts_list:
        print(f"\n{'─'*60}")
        print(f"N = {N} Sybil accounts  "
              f"(~{len(attacker_records) // N} queries/account)")
        print(f"{'─'*60}")

        sybil_records   = redistribute_queries(attacker_records, N)
        account_results = run_prada_on_records(sybil_records, delta)

        n_flagged = sum(1 for r in account_results.values() if r["flagged"])
        n_missed  = N - n_flagged   # accounts that escaped detection

        print(f"\n{'Account':<15} {'Queries':>8} {'W score':>10} {'Status'}")
        print("-" * 50)
        for acct in sorted(account_results):
            r  = account_results[acct]
            w  = f"{r['W']:.4f}" if r["W"] is not None else "N/A (warmup)"
            st = "FLAGGED" if r["flagged"] else "missed"
            print(f"{acct:<15} {r['n_queries']:>8} {w:>10}  {st}")

        print(f"\n  Flagged : {n_flagged}/{N}")
        print(f"  Missed  : {n_missed}/{N}")

        if n_flagged == 0:
            print("  *** PRADA COMPLETELY BLIND at this split level ***")

        experiment_results[N] = {
            "n_accounts":          N,
            "total_queries":       len(attacker_records),
            "queries_per_account": len(attacker_records) // N,
            "accounts_flagged":    n_flagged,
            "accounts_missed":     n_missed,
            "per_account":         account_results,
        }

    return experiment_results


# ---------------------------------------------------------------------------
# Summary table + failure analysis
# ---------------------------------------------------------------------------

def print_summary_table(results: dict) -> None:
    print("\n")
    print("=" * 70)
    print("RESULTS SUMMARY — PRADA Detection Rate vs Sybil Split Level")
    print("=" * 70)
    header = (f"{'Split Level':<14} {'Total Queries':>14} "
              f"{'Queries/Acct':>13} {'Flagged':>10} {'Missed':>10}")
    print(header)
    print("-" * 70)

    prada_blind_at = None

    for N, data in sorted(results.items()):
        split_label = f"{N} accounts"
        row = (
            f"{split_label:<14} "
            f"{data['total_queries']:>14,} "
            f"{data['queries_per_account']:>13,} "
            f"{data['accounts_flagged']:>10} "
            f"{data['accounts_missed']:>10}"
        )
        note = ""
        if data["accounts_flagged"] == 0 and prada_blind_at is None:
            note = "  ← PRADA blind"
            prada_blind_at = N
        print(row + note)

    print("=" * 70)

    # Failure analysis
    print("\nFailure Analysis")
    print("─" * 50)
    # Partial evasion: at least one account slipped through
    partial_evade = {N: d for N, d in results.items()
                     if d["accounts_missed"] > 0 and d["accounts_flagged"] > 0}
    if partial_evade:
        first_partial = min(partial_evade)
        pct = results[first_partial]["accounts_missed"] / first_partial * 100
        print(f"Partial evasion starts at N = {first_partial} accounts "
              f"({pct:.0f}% of accounts escape detection).")
        print(f"  Reason: W scores near δ threshold — statistical noise at the boundary.")

    if prada_blind_at is not None:
        d = results[prada_blind_at]
        warmup = sum(1 for r in d["per_account"].values() if r["W"] is None)
        if warmup > 0:
            print(f"\nComplete PRADA failure at N = {prada_blind_at} accounts.")
            print(f"  Each account receives ~{d['queries_per_account']} queries,")
            print(f"  but dmin distances per account fall below MIN_QUERIES={MIN_QUERIES}")
            print(f"  (first query per class generates no dmin → ~10 fewer values).")
            print(f"  Result: {warmup}/{prada_blind_at} accounts stuck in warmup, 0 flagged.")
        else:
            print(f"\nComplete PRADA failure at N = {prada_blind_at} accounts.")
            print(f"  Each account has W > δ — dmin sequence looks statistically normal.")
        print(f"\nConclusion: A Sybil attacker splitting across "
              f"{prada_blind_at}+ accounts evades PRADA completely.")
        print("This is the vulnerability our Sybil detection layer will close.")
    else:
        print("\nPRADA detected attacks at all tested split levels.")
        print("Try N ≥ 64 (6400 total queries / 64 = 100 q/acct) to find the warmup failure.")

    # Per-split W-score summary
    print("\nPer-Account W Scores (Shapiro-Wilk, threshold = FLAGGED if W < δ)")
    print("─" * 50)
    for N, data in sorted(results.items()):
        flagged_ws = [
            r["W"] for r in data["per_account"].values()
            if r["flagged"] and r["W"] is not None
        ]
        missed_ws = [
            r["W"] for r in data["per_account"].values()
            if not r["flagged"] and r["W"] is not None
        ]
        warmup_count = sum(
            1 for r in data["per_account"].values() if r["W"] is None
        )
        print(f"  N={N:>2}:  "
              f"flagged W range = {_range_str(flagged_ws)}  "
              f"missed W range = {_range_str(missed_ws)}  "
              f"warmup={warmup_count}")


def _range_str(vals: list) -> str:
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return f"{lo:.4f}"
    return f"[{lo:.4f} – {hi:.4f}]"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_sybil_experiment()
    print_summary_table(results)
    print("\n[Done] Sybil simulation complete.")
