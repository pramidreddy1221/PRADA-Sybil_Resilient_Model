"""
simulation/mixed.py — Mixed Attack Simulation

A smarter single-account attacker who mixes normal images
with synthetic JbDA queries to dilute the attack pattern.

Strategy:
  - Runs full 6-round JbDA attack like attacker_001
  - After each round, injects NORMAL_RATIO normal images
  - Goal: make dmin distribution look more natural
  - Tests whether Shapiro-Wilk can still detect this

Normal ratio: 30% normal, 70% synthetic per round
"""

import numpy as np
import torch
from config import DEVICE, ROUNDS, SEED_PER_CLASS, SAVE_PATH

from attacker.substitute_model import SubstituteCNN
from attacker.seed              import get_seed_samples
from attacker.query             import query_victim
from attacker.train             import train_substitute, evaluate_substitute
from attacker.augment           import jacobian_augment

ACCOUNT_ID   = "mixed_001"
NORMAL_RATIO = 0.30


def run_mixed_attack(ratio=0.30, account_id="mixed_001"):
    substitute = SubstituteCNN().to(DEVICE)

    # Phase 1: Seed — same as real attacker
    print(f"\n[Mixed Attack] Account: {account_id}")
    print(f"[Phase 1] Loading seed samples...")
    seed_images, _ = get_seed_samples(SEED_PER_CLASS)
    print(f"  Loaded {len(seed_images)} seed images")

    print(f"[Phase 1] Querying victim with seed samples...")
    seed_labels, _ = query_victim(seed_images, account_id=account_id)

    all_images = seed_images.copy()
    all_labels = seed_labels.copy()

    # Keep a pool of normal images to inject each round
    normal_pool = seed_images.copy()

    for round_num in range(1, ROUNDS + 1):
        print(f"\n[Round {round_num}/{ROUNDS}]")
        print(f"  Dataset size: {len(all_images)} samples")

        # Train substitute
        substitute = train_substitute(substitute, all_images, all_labels)
        agreement  = evaluate_substitute(substitute, all_images, all_labels)
        print(f"  Agreement: {agreement*100:.2f}%")

        # Generate synthetic samples
        synthetic_images = jacobian_augment(substitute, all_images, all_labels)
        n_total     = len(synthetic_images)
        n_synthetic = int(n_total * (1 - ratio))
        n_normal    = n_total - n_synthetic
        print(f"  Generated {n_total} synthetic samples")
        print(f"  Querying {n_synthetic} synthetic + {n_normal} normal ({ratio*100:.0f}% normal)")

        # Query only n_synthetic synthetic images
        synthetic_labels, _ = query_victim(synthetic_images[:n_synthetic], account_id=account_id)

        # Query n_normal images from normal pool
        indices      = np.random.choice(len(normal_pool), size=n_normal, replace=True)
        normal_batch = normal_pool[indices]
        normal_labels, _ = query_victim(normal_batch, account_id=account_id)

        # Add ALL synthetic images to training; labels from queried images
        all_images = np.concatenate([all_images, synthetic_images], axis=0)
        all_labels = all_labels + synthetic_labels + normal_labels

    torch.save(substitute.state_dict(), SAVE_PATH)
    print(f"\n[Done] Mixed attack complete")
    print(f"  Total queries sent: check logs/queries.jsonl for {account_id}")


if __name__ == "__main__":
    run_mixed_attack()