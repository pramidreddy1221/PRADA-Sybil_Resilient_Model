"""
simulation/lambda_sweep.py — FGSM Lambda Sweep

Runs nine full 6-round Papernot/JbDA attacks, one per lambda value.
Everything is identical to the baseline attack except the FGSM step size.

Lambdas tested: 8/255, 16/255, 25.5/255, 32/255, 40/255, 48/255, 56/255, 64/255, 128/255

Run with (server must be up at 127.0.0.1:8010):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python simulation/lambda_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.stats import shapiro

from config import DEVICE, ROUNDS, SEED_PER_CLASS, LOG_PATH, DELTA
from attacker.substitute_model import SubstituteCNN
from attacker.seed             import get_seed_samples
from attacker.query            import query_victim
from attacker.train            import train_substitute, evaluate_substitute
import attacker.augment as _augment_mod
from attacker.augment          import jacobian_augment
from defense.logs              import load_logs


SWEEP = [
    (8   / 255, "lambda_008"),
    (16  / 255, "lambda_016"),
    (25.5/ 255, "lambda_025"),
    (32  / 255, "lambda_032"),
    (40  / 255, "lambda_040"),
    (48  / 255, "lambda_048"),
    (56  / 255, "lambda_056"),
    (64  / 255, "lambda_064"),
    (128 / 255, "lambda_128"),
]

PRADA_ACCOUNTS = [
    "lambda_008", "lambda_016", "lambda_025",
    "lambda_032", "lambda_040", "lambda_048",
    "lambda_056", "lambda_064", "lambda_128",
    "benign_001",
]

LAMBDA_LABELS = {
    "lambda_008": "8/255",
    "lambda_016": "16/255",
    "lambda_025": "25.5/255",
    "lambda_032": "32/255",
    "lambda_040": "40/255",
    "lambda_048": "48/255",
    "lambda_056": "56/255",
    "lambda_064": "64/255",
    "lambda_128": "128/255",
    "benign_001": "N/A",
}


# ---------------------------------------------------------------------------
# Attack runner — identical to baseline except LAMBDA is patched per call
# ---------------------------------------------------------------------------

def run_papernot_attack(account_id: str, lam: float) -> None:
    """Full 6-round JbDA/FGSM attack with the given lambda."""
    # Patch the module-level LAMBDA so jacobian_augment uses this value
    _augment_mod.LAMBDA = lam

    substitute = SubstituteCNN().to(DEVICE)

    print(f"\n{'='*60}")
    print(f"  Account: {account_id}  |  lambda: {lam:.4f}  ({lam*255:.1f}/255)")
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

        synthetic_images = jacobian_augment(substitute, all_images, all_labels)
        print(f"  Generated {len(synthetic_images)} synthetic samples")

        synthetic_labels, _ = query_victim(synthetic_images, account_id=account_id)

        all_images = np.concatenate([all_images, synthetic_images], axis=0)
        all_labels = all_labels + synthetic_labels

    print(f"\n[Done] {account_id} complete — total images: {len(all_images)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    existing_accounts = {r["account_id"] for r in load_logs(LOG_PATH)}

    for lam, account_id in SWEEP:
        if account_id in existing_accounts:
            print(f"Skipping {account_id} — already in log")
            continue
        run_papernot_attack(account_id, lam)

    all_records = load_logs(LOG_PATH)

    print("\n" + "="*55)
    print("  PRADA Detection — Lambda Sweep")
    print("="*55)
    print(f"  {'Account':<15} {'Lambda':>10} {'W score':>10} {'Flagged'}")
    print("  " + "-"*50)

    for account_id in PRADA_ACCOUNTS:
        records = [r for r in all_records if r["account_id"] == account_id]

        if not records:
            continue

        vectors = [np.array(r["input_vector"], dtype=np.float32) for r in records]

        D = []
        for i in range(1, len(vectors)):
            dists = [np.linalg.norm(vectors[i] - vectors[j]) for j in range(i)]
            D.append(min(dists))

        if len(D) < 100:
            print(f"  {account_id:<15} {'N/A':>10} {'warmup':>10}")
            continue

        W, _ = shapiro(D)
        flagged = W < DELTA
        status = "YES" if flagged else "NO"
        print(f"  {account_id:<15} {LAMBDA_LABELS[account_id]:>10} {W:>10.4f}  {status}")

    print("="*55)
