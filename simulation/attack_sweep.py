"""
simulation/attack_sweep.py — FGSM / I-FGSM / MI-FGSM Attack Sweep

Runs three full 6-round Papernot (JbDA) attacks back-to-back, each using a
different augmentation function, then evaluates all accounts with PRADA and
prints a comparison table.

Accounts used:
  attacker_fgsm   — standard single-step FGSM  (jacobian_augment)
  attacker_ifgsm  — iterative FGSM              (jacobian_augment_ifgsm)
  attacker_mifgsm — momentum iterative FGSM     (jacobian_augment_mifgsm)

Pre-existing accounts (shown for reference):
  attacker_001    — original baseline attacker
  benign_001      — legitimate user

Run with (server must be up at 127.0.0.1:8010):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python simulation/attack_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from config import DEVICE, ROUNDS, SEED_PER_CLASS
from attacker.substitute_model import SubstituteCNN
from attacker.seed             import get_seed_samples
from attacker.query            import query_victim
from attacker.train            import train_substitute, evaluate_substitute
from attacker.augment          import (
    jacobian_augment,
    jacobian_augment_ifgsm,
    jacobian_augment_mifgsm,
)
from defense.prada import run_prada_on_records
from defense.logs  import load_logs
from config        import LOG_PATH


# ---------------------------------------------------------------------------
# Reusable attack runner
# ---------------------------------------------------------------------------

def run_papernot_attack(account_id: str, augment_fn) -> None:
    """Run a full 6-round Papernot/JbDA attack with the given augment function."""
    substitute = SubstituteCNN().to(DEVICE)

    print(f"\n{'='*60}")
    print(f"  Attack: {account_id}  |  augment: {augment_fn.__name__}")
    print(f"{'='*60}")

    print("[Phase 1] Loading seed samples...")
    seed_images, _ = get_seed_samples(SEED_PER_CLASS)
    print(f"  Loaded {len(seed_images)} seed images")

    print("[Phase 1] Querying victim with seed samples...")
    seed_labels, _ = query_victim(seed_images, account_id=account_id)

    all_images = seed_images.copy()
    all_labels = seed_labels.copy()

    for round_num in range(1, ROUNDS + 1):
        print(f"\n[Round {round_num}/{ROUNDS}]")
        print(f"  Dataset size: {len(all_images)} samples")

        substitute = train_substitute(substitute, all_images, all_labels)
        agreement  = evaluate_substitute(substitute, all_images, all_labels)
        print(f"  Agreement: {agreement*100:.2f}%")

        synthetic_images = augment_fn(substitute, all_images, all_labels)
        print(f"  Generated {len(synthetic_images)} synthetic samples")

        synthetic_labels, _ = query_victim(synthetic_images, account_id=account_id)

        all_images = np.concatenate([all_images, synthetic_images], axis=0)
        all_labels = all_labels + synthetic_labels

    print(f"\n[Done] {account_id} complete — total images: {len(all_images)}")


# ---------------------------------------------------------------------------
# PRADA results table
# ---------------------------------------------------------------------------

DISPLAY_ORDER = [
    "attacker_001",
    "attacker_fgsm",
    "attacker_ifgsm",
    "attacker_mifgsm",
    "benign_001",
]


def print_prada_table(prada_results: dict) -> None:
    print(f"\n\n{'='*50}")
    print("  PRADA Detection — Attack Sweep")
    print(f"{'='*50}")
    print(f"  {'Account':<20} {'W score':<12} {'Flagged'}")
    print("  " + "-" * 38)

    for acct in DISPLAY_ORDER:
        if acct not in prada_results:
            print(f"  {acct:<20} {'N/A':<12} N/A (no queries)")
            continue
        r = prada_results[acct]
        w_str      = f"{r['W']:.4f}" if r["W"] is not None else "N/A (warmup)"
        flagged_str = "YES" if r["flagged"] else "NO"
        print(f"  {acct:<20} {w_str:<12} {flagged_str}")

    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    attacks = [
        ("attacker_fgsm",   jacobian_augment),
        ("attacker_ifgsm",  jacobian_augment_ifgsm),
        ("attacker_mifgsm", jacobian_augment_mifgsm),
    ]

    existing_accounts = {r["account_id"] for r in load_logs(LOG_PATH)}

    for account_id, augment_fn in attacks:
        if account_id in existing_accounts:
            print(f"Skipping {account_id} — already in log")
            continue
        run_papernot_attack(account_id, augment_fn)

    # Load all logs and filter to the accounts we want to compare
    print("\n\n[PRADA] Loading query logs...")
    all_records = load_logs(LOG_PATH)
    sweep_records = [r for r in all_records if r["account_id"] in DISPLAY_ORDER]
    print(f"  {len(sweep_records)} records across {len(set(r['account_id'] for r in sweep_records))} accounts")

    prada_results = run_prada_on_records(sweep_records)
    print_prada_table(prada_results)

    import json
    out_path = _ROOT / "analysis" / "results" / "attack_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {acct: {"W": r["W"], "flagged": r["flagged"]} for acct, r in prada_results.items()}
    out_path.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path.relative_to(_ROOT)}")
