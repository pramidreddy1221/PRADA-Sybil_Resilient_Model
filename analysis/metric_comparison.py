import sys
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance as scipy_wasserstein
from scipy.spatial.distance import cosine as scipy_cosine

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defense.logs import load_logs
from defense.distances import compute_dmin_per_account
from defense.sybil_detection import js_divergence
from simulation.sybil import redistribute_queries
from config import LOG_PATH, SYBIL_MIN_DMIN, SYBIL_N_BINS

N_SYBIL = 64
BENIGN_ID = "benign_001"


def kl_symmetric(p: np.ndarray, q: np.ndarray) -> float:
    kl_pq = np.sum(p * np.log(p / (q + 1e-300) + 1e-300))
    kl_qp = np.sum(q * np.log(q / (p + 1e-300) + 1e-300))
    return float(np.clip(0.5 * (kl_pq + kl_qp), 0.0, None))


def wasserstein(p: np.ndarray, q: np.ndarray, bin_centers: np.ndarray) -> float:
    return float(scipy_wasserstein(bin_centers, bin_centers, u_weights=p, v_weights=q))


def cosine_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(scipy_cosine(p, q))


def build_histograms_with_bins(
    dmin_per_account: dict,
    n_bins: int = SYBIL_N_BINS,
) -> tuple[dict, np.ndarray]:
    all_values = [v for D in dmin_per_account.values() for v in D]
    if not all_values:
        return {}, np.array([])

    global_min = float(min(all_values))
    global_max = float(max(all_values))
    if global_min == global_max:
        global_max = global_min + 1e-6

    bins = np.linspace(global_min, global_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    histograms = {}
    for account_id, D in dmin_per_account.items():
        hist, _ = np.histogram(D, bins=bins)
        hist = hist.astype(float) + 1e-10
        hist /= hist.sum()
        histograms[account_id] = hist

    return histograms, bin_centers


def compute_pairwise_matrix(histograms: dict, metric_fn) -> tuple[list, np.ndarray]:
    accounts = sorted(histograms.keys())
    n = len(accounts)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = metric_fn(histograms[accounts[i]], histograms[accounts[j]])
            matrix[i, j] = d
            matrix[j, i] = d
    return accounts, matrix


def separation_stats(
    accounts: list, matrix: np.ndarray, benign_id: str
) -> tuple[float, float, float, bool]:
    sybil_idx = [i for i, a in enumerate(accounts) if a != benign_id]
    benign_idx = [i for i, a in enumerate(accounts) if a == benign_id]

    within_vals = [
        matrix[i, j]
        for idx, i in enumerate(sybil_idx)
        for j in sybil_idx[idx + 1:]
    ]
    cross_vals = [
        matrix[i, j]
        for i in sybil_idx
        for j in benign_idx
    ]

    within_mean = float(np.mean(within_vals)) if within_vals else 0.0
    cross_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
    gap = cross_mean - within_mean
    return within_mean, cross_mean, gap, gap > 0


def main() -> None:
    all_records = load_logs(LOG_PATH)
    attacker_records = [r for r in all_records if r["account_id"] == "attacker_001"][:6400]
    benign_records = [r for r in all_records if r["account_id"] == BENIGN_ID][:3000]

    if not attacker_records:
        print("ERROR: No attacker_001 records in log.")
        sys.exit(1)
    if not benign_records:
        print("ERROR: No benign_001 records in log.")
        sys.exit(1)

    print(f"attacker_001: {len(attacker_records)} queries")
    print(f"benign_001:   {len(benign_records)} queries")

    sybil_records = redistribute_queries(attacker_records, N_SYBIL)
    mixed_records = sybil_records + benign_records
    print(f"Redistributed into {N_SYBIL} Sybil accounts (round-robin) + 1 benign\n")

    account_dmin_data = compute_dmin_per_account(mixed_records)
    eligible = {
        acct: data["D"]
        for acct, data in account_dmin_data.items()
        if len(data["D"]) >= SYBIL_MIN_DMIN
    }

    sybil_eligible = [a for a in eligible if a != BENIGN_ID]
    benign_eligible = [a for a in eligible if a == BENIGN_ID]
    print(f"Eligible accounts (>= {SYBIL_MIN_DMIN} dmin values):")
    print(f"  Sybil : {len(sybil_eligible)}")
    print(f"  Benign: {len(benign_eligible)}")

    if not benign_eligible:
        print(f"\nWARNING: {BENIGN_ID} has fewer than {SYBIL_MIN_DMIN} dmin values "
              f"— cross-group comparison will be empty.")

    histograms, bin_centers = build_histograms_with_bins(eligible)
    if not histograms:
        print("ERROR: No eligible accounts.")
        sys.exit(1)

    metrics = [
        ("JS",          lambda p, q, _bc=bin_centers: js_divergence(p, q)),
        ("KL",          lambda p, q, _bc=bin_centers: kl_symmetric(p, q)),
        ("Wasserstein", lambda p, q, bc=bin_centers:  wasserstein(p, q, bc)),
        ("Cosine",      lambda p, q, _bc=bin_centers: cosine_dist(p, q)),
    ]

    print()
    hdr = f"{'Metric':<14} {'Within-Sybil':>13} {'Sybil-Benign':>13} {'Gap':>10} {'Separates?':>10}"
    print(hdr)
    print("-" * len(hdr))

    metric_rows = []
    for name, fn in metrics:
        accounts, matrix = compute_pairwise_matrix(histograms, fn)
        within, cross, gap, separates = separation_stats(accounts, matrix, BENIGN_ID)
        print(
            f"{name:<14} {within:>13.4f} {cross:>13.4f} {gap:>10.4f}"
            f" {'YES' if separates else 'NO':>10}"
        )
        metric_rows.append({
            "metric": name, "within_mean": within,
            "cross_mean": cross, "gap": gap, "separates": separates,
        })

    print()

    import json
    out_path = _ROOT / "analysis" / "results" / "metric_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    print(f"Saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
