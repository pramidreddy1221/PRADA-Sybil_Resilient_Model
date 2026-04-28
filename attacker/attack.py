from __future__ import annotations

import json
import numpy as np
import torch
from config import DEVICE, ROUNDS, SEED_PER_CLASS, SAVE_PATH, RESULTS_PATH, ROOT

from attacker.substitute_model import SubstituteCNN
from attacker.seed              import get_seed_samples
from attacker.query             import query_victim
from attacker.train             import train_substitute, evaluate_substitute
from attacker.augment           import jacobian_augment


def run_attack():
    substitute = SubstituteCNN().to(DEVICE)

    print("\n[Phase 1] Loading seed samples...")
    seed_images, _ = get_seed_samples(SEED_PER_CLASS)
    print(f"  Loaded {len(seed_images)} seed images")

    print("\n[Phase 1] Querying victim API with seed samples...")
    seed_labels, _ = query_victim(seed_images)
    print(f"  Received {len(seed_labels)} labels from victim")

    all_images = seed_images.copy()
    all_labels = seed_labels.copy()

    results = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n[Round {round_num}/{ROUNDS}]")
        print(f"  Dataset size: {len(all_images)} samples")

        print(f"  Training substitute model...")
        substitute = train_substitute(substitute, all_images, all_labels)

        agreement = evaluate_substitute(substitute, all_images, all_labels)
        print(f"  Agreement with victim: {agreement*100:.2f}%")

        results.append({
            "round": round_num,
            "n_samples": len(all_images),
            "agreement": round(agreement, 4)
        })

        print(f"  Generating synthetic samples (FGSM)...")
        synthetic_images = jacobian_augment(substitute, all_images, all_labels)
        print(f"  Generated {len(synthetic_images)} synthetic samples")

        print(f"  Querying victim with synthetic samples...")
        synthetic_labels, _ = query_victim(synthetic_images)

        all_images = np.concatenate([all_images, synthetic_images], axis=0)
        all_labels = all_labels + synthetic_labels

    torch.save(substitute.state_dict(), SAVE_PATH)
    print(f"\n[Done] Substitute model saved -> {SAVE_PATH}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Results saved -> {RESULTS_PATH}")

    print("\n[Summary]")
    print(f"{'Round':<8} {'Samples':<10} {'Agreement'}")
    for r in results:
        print(f"{r['round']:<8} {r['n_samples']:<10} {r['agreement']*100:.2f}%")


def run_attack_cvsearch():
    from attacker.train import train_substitute_cvsearch, train_substitute_fixed
    from attacker.substitute_model_cv import SubstituteCNNWithDropout

    substitute = SubstituteCNNWithDropout().to(DEVICE)

    print("\n[Phase 1] Loading seed samples...")
    seed_images, _ = get_seed_samples(SEED_PER_CLASS)
    print(f"  Loaded {len(seed_images)} seed images")

    print("\n[Phase 1] Querying victim API with seed samples...")
    seed_labels, _ = query_victim(seed_images, account_id="attacker_cvsearch")
    print(f"  Received {len(seed_labels)} labels from victim")

    all_images = seed_images.copy()
    all_labels = seed_labels.copy()

    print("\n[CV Search] Finding best hyperparameters on seed data...")
    substitute, best_lr, best_epochs = train_substitute_cvsearch(
        substitute, all_images, all_labels
    )
    print(f"  → lr={best_lr:.0e}  epochs={best_epochs}")

    results = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n[Round {round_num}/{ROUNDS}]")
        print(f"  Dataset size: {len(all_images)} samples")

        print(f"  Training substitute model (lr={best_lr:.0e}, epochs={best_epochs})...")
        substitute = train_substitute_fixed(
            substitute, all_images, all_labels, best_lr, best_epochs
        )

        agreement = evaluate_substitute(substitute, all_images, all_labels)
        print(f"  Agreement with victim: {agreement*100:.2f}%")

        results.append({
            "round": round_num,
            "n_samples": len(all_images),
            "agreement": round(agreement, 4)
        })

        print(f"  Generating synthetic samples (Jacobian augmentation)...")
        synthetic_images = jacobian_augment(substitute, all_images, all_labels)
        print(f"  Generated {len(synthetic_images)} synthetic samples")

        print(f"  Querying victim with synthetic samples...")
        synthetic_labels, _ = query_victim(synthetic_images, account_id="attacker_cvsearch")

        all_images = np.concatenate([all_images, synthetic_images], axis=0)
        all_labels = all_labels + synthetic_labels

    cvsearch_results_path = ROOT / "attacker" / "cvsearch_results.json"

    cvsearch_model_path = SAVE_PATH.parent / "substitute_model_cv.pt"
    torch.save(substitute.state_dict(), cvsearch_model_path)
    print(f"\n[Done] CV-Search model saved -> {cvsearch_model_path}")

    with open(cvsearch_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Results saved -> {cvsearch_results_path}")

    print("\n[Summary]")
    print(f"{'Round':<8} {'Samples':<10} {'Agreement'}")
    for r in results:
        print(f"{r['round']:<8} {r['n_samples']:<10} {r['agreement']*100:.2f}%")


if __name__ == "__main__":
    run_attack()
