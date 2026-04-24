"""
simulation/mixed_sweep.py — Sweep mixed attack ratios through PRADA detection.

Runs the mixed attacker at six normal-query injection ratios, then evaluates
each account with PRADA Shapiro-Wilk detection and prints a comparison table
alongside the pure-attack and pure-benign baselines.

Run with:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python simulation/mixed_sweep.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.mixed import run_mixed_attack
from defense.prada import run_prada_on_records
from defense.logs import load_logs
from config import LOG_PATH

RATIOS = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
ACCOUNT_IDS = [
    "mixed_010",
    "mixed_020",
    "mixed_030",
    "mixed_050",
    "mixed_070",
    "mixed_090",
]


def run_sweep():
    print("=" * 62)
    print("  Mixed Attack Ratio Sweep")
    print("  Ratios: " + ", ".join(f"{r:.0%}" for r in RATIOS))
    print("=" * 62)

    # ------------------------------------------------------------------ #
    # Phase 1: Run all attacks sequentially                                #
    # ------------------------------------------------------------------ #
    existing_accounts = {r["account_id"] for r in load_logs(LOG_PATH)}

    for ratio, account_id in zip(RATIOS, ACCOUNT_IDS):
        if account_id in existing_accounts:
            print(f"Skipping {account_id} — already in log")
            continue
        print(f"\n{'─'*62}")
        print(f"  ratio={ratio:.2f}  ({ratio*100:.0f}% normal / {(1-ratio)*100:.0f}% synthetic)"
              f"  →  {account_id}")
        print(f"{'─'*62}")
        run_mixed_attack(ratio=ratio, account_id=account_id)

    # ------------------------------------------------------------------ #
    # Phase 2 + 3: Run PRADA and print table                               #
    # ------------------------------------------------------------------ #
    ACCOUNTS = ["mixed_010", "mixed_020", "mixed_030",
                "mixed_050", "mixed_070", "mixed_090",
                "attacker_001", "benign_001"]

    all_records = load_logs(LOG_PATH)
    filtered    = [r for r in all_records if r["account_id"] in ACCOUNTS]
    results     = run_prada_on_records(filtered)

    print(f"\n\n{'='*62}")
    print("  PRADA Detection — Normal Ratio Sweep")
    print(f"{'='*62}")
    print(f"  {'Account':<14}  {'Normal%':>8}  {'Synth%':>7}  {'W score':>8}  Flagged")
    print("  " + "─" * 56)

    for ratio, account_id in zip(RATIOS, ACCOUNT_IDS):
        res        = results.get(account_id, {})
        w          = res.get("W")
        flagged    = res.get("flagged", False)
        w_str      = f"{w:.4f}" if w is not None else "warmup"
        flag_str   = "YES" if flagged else "NO"
        normal_pct = f"{ratio*100:.0f}%"
        synth_pct  = f"{(1-ratio)*100:.0f}%"
        print(f"  {account_id:<14}  {normal_pct:>8}  {synth_pct:>7}  {w_str:>8}  {flag_str}")

    # Reference rows
    print("  " + "─" * 56)

    r_att    = results.get("attacker_001", {})
    w_att_s  = f"{r_att['W']:.4f}" if r_att.get("W") is not None else "N/A"
    flag_att = "YES" if r_att.get("flagged", False) else "NO"
    print(f"  {'attacker_001':<14}  {'0%':>8}  {'100%':>7}  {w_att_s:>8}  {flag_att}"
          "  ← pure attack baseline")

    r_ben    = results.get("benign_001", {})
    w_ben_s  = f"{r_ben['W']:.4f}" if r_ben.get("W") is not None else "N/A"
    flag_ben = "YES" if r_ben.get("flagged", False) else "NO"
    print(f"  {'benign_001':<14}  {'100%':>8}  {'0%':>7}  {w_ben_s:>8}  {flag_ben}"
          "  ← pure benign baseline")

    print(f"{'='*62}\n")


if __name__ == "__main__":
    run_sweep()
