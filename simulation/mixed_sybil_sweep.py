import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.mixed import run_mixed_attack
from simulation.sybil import redistribute_queries
from defense.prada import run_prada_on_records
from defense.sybil_detection import run_sybil_detection
from defense.logs import load_logs
from config import LOG_PATH

N_SYBIL = 64
SOURCE_ACCOUNT = "mixed_sybil_source"
BENIGN_ACCOUNT = "benign_001"
MIXED_RATIO = 0.30


def main() -> None:
    print("Mixed Sybil Sweep")
    print(f"  ratio={MIXED_RATIO}  N={N_SYBIL} Sybil accounts")

    existing = {r["account_id"] for r in load_logs(LOG_PATH)}
    if SOURCE_ACCOUNT not in existing:
        print(f"\n[Step 1] Running mixed attack as '{SOURCE_ACCOUNT}'...")
        run_mixed_attack(ratio=MIXED_RATIO, account_id=SOURCE_ACCOUNT)
    else:
        print(f"\n[Step 1] Skipping attack — {SOURCE_ACCOUNT} already in log")

    print(f"\n[Step 2] Loading records from log...")
    all_records = load_logs(LOG_PATH)
    source_records = [r for r in all_records if r["account_id"] == SOURCE_ACCOUNT][:6400]
    benign_records = [r for r in all_records if r["account_id"] == BENIGN_ACCOUNT][:3000]

    print(f"  {SOURCE_ACCOUNT} : {len(source_records)} records")
    print(f"  {BENIGN_ACCOUNT}   : {len(benign_records)} records")

    if not source_records:
        print(f"\n[ERROR] No records for '{SOURCE_ACCOUNT}'. Did the attack reach the server?")
        sys.exit(1)

    print(f"\n[Step 3] Redistributing across {N_SYBIL} Sybil accounts (round-robin)...")
    sybil_records = redistribute_queries(source_records, N_SYBIL)
    per_account = len(source_records) // N_SYBIL
    print(f"  ~{per_account} queries/account")

    if benign_records:
        mixed_records = sybil_records + benign_records
        print(f"\n[Step 4] Mixed dataset: {N_SYBIL} Sybil + '{BENIGN_ACCOUNT}'")
    else:
        mixed_records = sybil_records
        print(f"\n[Step 4] '{BENIGN_ACCOUNT}' not in log — running Sybil-only (no FP check)")

    print(f"\n[Step 5a] PRADA per-account detection...")
    prada_results = run_prada_on_records(mixed_records)

    sybil_accts = sorted(a for a in prada_results if a.startswith("sybil_"))
    benign_accts = sorted(a for a in prada_results if not a.startswith("sybil_"))

    n_sybil_flagged = sum(1 for a in sybil_accts if prada_results[a]["flagged"])
    n_sybil_warmup = sum(1 for a in sybil_accts if prada_results[a]["W"] is None)
    n_benign_flagged = sum(1 for a in benign_accts if prada_results[a]["flagged"])

    print(f"\n[Step 5b] JS divergence cross-account detection...")
    sd = run_sybil_detection(mixed_records, verbose=False)

    sybil_in_cluster = [a for a in sd["flagged_accounts"] if a.startswith("sybil_")]
    benign_in_cluster = [a for a in sd["flagged_accounts"] if not a.startswith("sybil_")]

    benign_fp = bool(benign_in_cluster) or bool(n_benign_flagged)

    print("RESULTS")

    warmup_note = f"  ({n_sybil_warmup} in warmup — below MIN_QUERIES)" if n_sybil_warmup else ""
    print(f"PRADA:        {n_sybil_flagged}/{N_SYBIL} accounts flagged{warmup_note}")

    js_note = f"  ({sd['cluster_size']} accounts in cluster)"
    print(f"JS detection: sybil detected {'YES' if sd['sybil_detected'] else 'NO'}{js_note}")

    if benign_records:
        fp_detail = ""
        if benign_in_cluster:
            fp_detail = "  (caught by JS detector)"
        elif n_benign_flagged:
            fp_detail = "  (caught by PRADA)"
        print(f"False positives: benign flagged {'YES' if benign_fp else 'NO'}{fp_detail}")
    else:
        print(f"False positives: N/A  (no '{BENIGN_ACCOUNT}' records in log)")

    if sd["mean_js_within"] is not None or sd["mean_js_cross"] is not None:
        print()
        print("JS diagnostics:")
        if sd["mean_js_within"] is not None:
            print(f"  Within-Sybil mean JS : {sd['mean_js_within']:.4f}")
        if sd["mean_js_cross"] is not None:
            print(f"  Sybil-Benign mean JS : {sd['mean_js_cross']:.4f}")

    out_path = _ROOT / "analysis" / "results" / "mixed_sybil_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_data = {
        "source_account": SOURCE_ACCOUNT,
        "n_sybil": N_SYBIL,
        "prada_flagged": n_sybil_flagged,
        "prada_warmup": n_sybil_warmup,
        "js_detected": sd["sybil_detected"],
        "js_cluster_size": sd["cluster_size"],
        "benign_fp": benign_fp,
        "mean_js_within": sd["mean_js_within"],
        "mean_js_cross": sd["mean_js_cross"],
    }
    out_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
