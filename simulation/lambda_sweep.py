from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

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


def run_papernot_attack(account_id: str, lam: float) -> None:
    _augment_mod.LAMBDA = lam  # overrides the module-level LAMBDA constant so jacobian_augment picks up the sweep value at call time without reimporting

    substitute = SubstituteCNN().to(DEVICE)

    print(f"\n  Account: {account_id}  |  lambda: {lam:.4f}  ({lam*255:.1f}/255)")

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
        agreement = evaluate_substitute(substitute, all_images, all_labels)
        print(f"  Agreement: {agreement*100:.2f}%")

        synthetic_images = jacobian_augment(substitute, all_images, all_labels)
        print(f"  Generated {len(synthetic_images)} synthetic samples")

        synthetic_labels, _ = query_victim(synthetic_images, account_id=account_id)

        all_images = np.concatenate([all_images, synthetic_images], axis=0)
        all_labels = all_labels + synthetic_labels

    print(f"\n[Done] {account_id} complete — total images: {len(all_images)}")


if __name__ == "__main__":
    existing_accounts = {r["account_id"] for r in load_logs(LOG_PATH)}

    for lam, account_id in SWEEP:
        if account_id in existing_accounts:
            print(f"Skipping {account_id} — already in log")
            continue
        run_papernot_attack(account_id, lam)
