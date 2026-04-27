from config import DELTA, LOG_PATH
from defense.logs import load_logs
from defense.distances import compute_dmin_per_account
from defense.detection import run_shapiro


def run_prada_on_records(records: list, delta: float = DELTA) -> dict:
    account_dmin_data = compute_dmin_per_account(records)
    results = {}
    for acct, data in account_dmin_data.items():
        sh = run_shapiro(data["D"], delta)
        results[acct] = {
            "n_queries": data["n_queries"],
            "n_distances": len(data["D"]),
            "W": sh["W"],
            "p_value": sh["p_value"],
            "flagged": sh["flagged"],
            "reason": sh["reason"],
        }
    return results


def run_prada(delta: float = DELTA, log_path=LOG_PATH) -> dict:
    print("PRADA Detection (Algorithm 3)")
    print(f"δ threshold: {delta}")

    records = load_logs(log_path)
    print(f"\nLoaded {len(records)} query records")

    account_results = compute_dmin_per_account(records)

    final_results = {}

    print(f"\n{'Account':<20} {'Queries':<10} {'Distances':<12} {'W':<8} {'Flagged'}")
    print("-" * 60)
    for account_id, data in account_results.items():
        D = data["D"]
        n_queries = data["n_queries"]

        shapiro_result = run_shapiro(D, delta)

        final_results[account_id] = {
            "n_queries": n_queries,
            "n_distances": len(D),
            "W": shapiro_result["W"],
            "p_value": shapiro_result["p_value"],
            "flagged": shapiro_result["flagged"],
            "reason": shapiro_result["reason"]
        }

        W_str = str(shapiro_result["W"]) if shapiro_result["W"] else "N/A"
        flagged_str = "🚨 ATTACK" if shapiro_result["flagged"] else "✅ benign"

        print(f"{account_id:<20} {n_queries:<10} {len(D):<12} {W_str:<8} {flagged_str}")

    return final_results

if __name__ == "__main__":
    results = run_prada()
    print("\n[Done] PRADA detection complete")
