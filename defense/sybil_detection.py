"""
defense/sybil_detection.py — Cross-Account Sybil Detection

The core idea (plain English):
  Imagine 64 bank robbers each making only 100 small transactions, so
  the per-person fraud detector never triggers.  But all 64 of them are
  doing transactions that look almost identical — same amounts, same
  timing patterns.  If you compare them side-by-side, the coordinated
  fraud is obvious.

  That is exactly what this module does.  Instead of watching each
  account alone, it compares the dmin distance distributions across
  accounts.  Attackers using the same FGSM-based Papernot strategy
  produce nearly identical dmin sequences.  Benign users look different
  from one another.  We catch the attack by measuring pairwise
  distribution similarity (Jensen-Shannon divergence) across accounts.

Algorithm (Algorithm 4 — Sybil Detection Layer):
  1. Compute dmin sequences per account (reuses distances.py).
  2. For each account with >= SYBIL_MIN_DMIN values, build a normalised
     probability histogram using shared bin edges (global min/max).
  3. Compute the full pairwise Jensen-Shannon divergence matrix.
  4. An account is a Sybil candidate if it has at least
     (SYBIL_MIN_CLUSTER - 1) "similar" neighbours (JS < SYBIL_JS_THRESHOLD).
  5. If the flagged cluster has >= SYBIL_MIN_CLUSTER members, report a
     coordinated Sybil attack.

Jensen-Shannon Divergence:
  A symmetric, bounded similarity measure between two probability
  distributions.  Range: [0, ln(2)] ≈ [0, 0.693].
    JS = 0     → identical distributions
    JS = ln(2) → maximally different
  We use JS rather than KL because JS is always finite and symmetric.
"""

import sys
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defense.distances import compute_dmin_per_account
from config import (
    SYBIL_MIN_DMIN,
    SYBIL_JS_THRESHOLD,
    SYBIL_MIN_CLUSTER,
    SYBIL_N_BINS,
    LOG_PATH,
)
from defense.logs import load_logs


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two probability distributions p and q.

    Both arrays must be non-negative and already sum to 1.
    The midpoint m = (p+q)/2 acts as the "average" distribution.
    JS(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m)

    The 1e-300 guard prevents log(0) even if smoothing was imperfect.
    Returns a value in [0, ~0.693].
    """
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / (m + 1e-300) + 1e-300))
    kl_qm = np.sum(q * np.log(q / (m + 1e-300) + 1e-300))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, None))


# ---------------------------------------------------------------------------
# Histogram construction
# ---------------------------------------------------------------------------

def build_histograms(
    dmin_per_account: dict,
    n_bins: int = SYBIL_N_BINS,
) -> dict:
    """
    Build a normalised histogram for each account's dmin distribution,
    using shared bin edges derived from the global min/max.

    Why shared bins?  Without them, two accounts with the same dmin
    distribution but slightly different ranges would appear different —
    shared bins make the comparison apples-to-apples.

    Laplace smoothing (+1e-10 per bin) prevents log(0) in JS computation.

    Args:
        dmin_per_account: {account_id: [dmin_value, ...]}
        n_bins:           number of histogram bins

    Returns:
        {account_id: probability_array (shape: [n_bins])}
    """
    all_values = [v for D in dmin_per_account.values() for v in D]

    if not all_values:
        return {}

    global_min = float(min(all_values))
    global_max = float(max(all_values))

    # Edge case: all values identical (degenerate distribution)
    if global_min == global_max:
        global_max = global_min + 1e-6

    bins = np.linspace(global_min, global_max, n_bins + 1)

    histograms = {}
    for account_id, D in dmin_per_account.items():
        hist, _ = np.histogram(D, bins=bins)
        hist = hist.astype(float) + 1e-10   # Laplace smoothing
        hist /= hist.sum()                   # normalise → probability distribution
        histograms[account_id] = hist

    return histograms


# ---------------------------------------------------------------------------
# Pairwise JS matrix
# ---------------------------------------------------------------------------

def compute_pairwise_js(histograms: dict) -> tuple:
    """
    Compute the full pairwise Jensen-Shannon divergence matrix.

    Returns:
        (accounts_list, n×n JS matrix)   — matrix is symmetric, diagonal=0.
    """
    accounts = sorted(histograms.keys())
    n = len(accounts)
    js_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            js = js_divergence(histograms[accounts[i]], histograms[accounts[j]])
            js_matrix[i, j] = js
            js_matrix[j, i] = js

    return accounts, js_matrix


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def find_sybil_cluster(
    accounts: list,
    js_matrix: np.ndarray,
    js_threshold: float = SYBIL_JS_THRESHOLD,
    min_cluster: int = SYBIL_MIN_CLUSTER,
) -> list:
    """
    Identify accounts that belong to a Sybil cluster.

    An account is flagged if it has at least (min_cluster - 1) neighbours
    whose JS divergence from it is below js_threshold.

    Analogy: you're suspicious not just if you resemble ONE other person,
    but if you're part of a GROUP all behaving nearly identically.

    Returns:
        List of account IDs flagged as Sybil members.
    """
    n = len(accounts)
    flagged = []

    for i in range(n):
        n_similar = sum(
            1 for j in range(n)
            if j != i and js_matrix[i, j] < js_threshold
        )
        if n_similar >= min_cluster - 1:
            flagged.append(accounts[i])

    return flagged


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def run_sybil_detection(
    records: list,
    min_dmin: int = SYBIL_MIN_DMIN,
    js_threshold: float = SYBIL_JS_THRESHOLD,
    min_cluster: int = SYBIL_MIN_CLUSTER,
    n_bins: int = SYBIL_N_BINS,
    verbose: bool = True,
) -> dict:
    """
    Full cross-account Sybil detection pipeline.

    Args:
        records:      List of query dicts (same format as queries.jsonl).
                      Each dict must have: account_id, input_vector, pred.
        min_dmin:     Minimum dmin values per account to participate.
        js_threshold: JS divergence below this → two accounts are "similar".
        min_cluster:  Minimum accounts in a similar group to flag as Sybil.
        n_bins:       Histogram bin count for dmin distributions.
        verbose:      Print progress and per-account table.

    Returns dict:
        sybil_detected    (bool)
        flagged_accounts  (list[str])
        n_eligible        (int)   — accounts with enough dmin data
        cluster_size      (int)   — number of accounts flagged
        mean_js_within    (float|None) — mean JS inside the Sybil cluster
        mean_js_cross     (float|None) — mean JS between Sybil and non-Sybil
        js_matrix         (np.ndarray|None)
        accounts          (list[str])  — eligible accounts in matrix order
        reason            (str)
    """
    # Step 1 — compute dmin per account
    account_dmin_data = compute_dmin_per_account(records)

    # Step 2 — filter to accounts with enough dmin values
    eligible_dmins = {
        acct: data["D"]
        for acct, data in account_dmin_data.items()
        if len(data["D"]) >= min_dmin
    }

    n_eligible = len(eligible_dmins)

    if n_eligible < min_cluster:
        result = {
            "sybil_detected":   False,
            "flagged_accounts": [],
            "n_eligible":       n_eligible,
            "cluster_size":     0,
            "mean_js_within":   None,
            "mean_js_cross":    None,
            "js_matrix":        None,
            "accounts":         sorted(eligible_dmins.keys()),
            "reason":           f"not enough eligible accounts ({n_eligible} < {min_cluster})",
        }
        if verbose:
            print(f"    Not enough eligible accounts ({n_eligible} < {min_cluster}) — cannot run cross-account analysis.")
        return result

    # Step 3 — build shared-bin histograms
    histograms = build_histograms(eligible_dmins, n_bins)

    # Step 4 — pairwise JS matrix
    accounts, js_matrix = compute_pairwise_js(histograms)

    # Step 5 — detect Sybil cluster
    flagged = find_sybil_cluster(accounts, js_matrix, js_threshold, min_cluster)

    sybil_detected = len(flagged) >= min_cluster

    # Diagnostics: mean JS within cluster and cross-cluster
    mean_js_within = None
    mean_js_cross  = None

    if flagged:
        flagged_idx     = [accounts.index(a) for a in flagged]
        non_flagged_idx = [accounts.index(a) for a in accounts if a not in flagged]

        if len(flagged_idx) > 1:
            within_pairs = [
                js_matrix[i, j]
                for i in flagged_idx
                for j in flagged_idx
                if i < j
            ]
            if within_pairs:
                mean_js_within = float(np.mean(within_pairs))

        if non_flagged_idx:
            cross_pairs = [
                js_matrix[i, j]
                for i in flagged_idx
                for j in non_flagged_idx
            ]
            if cross_pairs:
                mean_js_cross = float(np.mean(cross_pairs))

    if verbose:
        _print_results(
            accounts, js_matrix, flagged,
            eligible_dmins, account_dmin_data,
            js_threshold, min_cluster,
            mean_js_within, mean_js_cross,
        )

    return {
        "sybil_detected":   sybil_detected,
        "flagged_accounts": flagged,
        "n_eligible":       n_eligible,
        "cluster_size":     len(flagged),
        "mean_js_within":   mean_js_within,
        "mean_js_cross":    mean_js_cross,
        "js_matrix":        js_matrix,
        "accounts":         accounts,
        "reason":           "coordinated attack detected" if sybil_detected else "no sybil cluster found",
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_results(
    accounts, js_matrix, flagged,
    eligible_dmins, account_dmin_data,
    js_threshold, min_cluster,
    mean_js_within, mean_js_cross,
):
    n = len(accounts)

    print(f"    Eligible accounts : {n}  (>= {SYBIL_MIN_DMIN} dmin values)")
    print(f"    JS threshold      : {js_threshold}")
    print(f"    Min cluster size  : {min_cluster}")
    print()

    # Cap display at 20 accounts to keep output readable
    display_accounts = accounts[:20]
    truncated = n > 20

    print(f"    {'Account':<22} {'dmin vals':>10} {'Mean JS':>10} {'Neighbors':>10} {'Status'}")
    print("    " + "-" * 58)

    for i, acct in enumerate(display_accounts):
        n_dmin = len(eligible_dmins[acct])
        others_js = [js_matrix[i, j] for j in range(n) if j != i]
        mean_js   = float(np.mean(others_js)) if others_js else 0.0
        n_neighbors = sum(
            1 for j in range(n) if j != i and js_matrix[i, j] < js_threshold
        )
        status = "SYBIL" if acct in flagged else "ok"
        print(f"    {acct:<22} {n_dmin:>10} {mean_js:>10.4f} {n_neighbors:>10} {status}")

    if truncated:
        n_sybil_hidden  = sum(1 for a in accounts[20:] if a in flagged)
        n_ok_hidden     = sum(1 for a in accounts[20:] if a not in flagged)
        print(f"    ... ({n - 20} more accounts: {n_sybil_hidden} SYBIL, {n_ok_hidden} ok)")

    print()
    if flagged:
        print(f"    Sybil cluster DETECTED: {len(flagged)} accounts flagged")
        if mean_js_within is not None:
            print(f"      Within-cluster mean JS : {mean_js_within:.4f}  (lower = more coordinated)")
        if mean_js_cross is not None:
            print(f"      Cross-group mean JS    : {mean_js_cross:.4f}  (higher = well separated)")
    else:
        print("    No Sybil cluster detected — accounts appear independent.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records = load_logs(LOG_PATH)
    print("=" * 60)
    print("Sybil Detection — Cross-Account Analysis")
    print(f"Loaded {len(records)} records from {LOG_PATH.name}")
    print("=" * 60)
    result = run_sybil_detection(records, verbose=True)
    print(f"\nResult: sybil_detected={result['sybil_detected']}")
    print(f"        flagged={len(result['flagged_accounts'])} accounts")
    print(f"        reason='{result['reason']}'")
