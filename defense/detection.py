import numpy as np
from scipy.stats import shapiro
from config import DELTA, MIN_QUERIES

def run_shapiro(D: list[float], delta: float = DELTA) -> dict:
    if len(D) < MIN_QUERIES:
        return {
            "W": None,
            "p_value": None,
            "flagged": False,
            "reason": f"warmup ({len(D)}/{MIN_QUERIES} queries)"
        }

    D_arr = np.array(D, dtype=np.float64)
    mean = np.mean(D_arr)
    std = np.std(D_arr)
    D_clean = D_arr[np.abs(D_arr - mean) <= 3 * std]  # 3-sigma clip removes outliers before Shapiro-Wilk; standard preprocessing for normality tests

    if len(D_clean) < 10:
        return {
            "W": None,
            "p_value": None,
            "flagged": False,
            "reason": "not enough data after outlier removal"
        }

    W, p_value = shapiro(D_clean)
    flagged = float(W) < delta  # flagged on W score, not p-value — matches Algorithm 3 in the paper

    return {
        "W": round(float(W), 4),
        "p_value": round(float(p_value), 4),
        "flagged": flagged,
        "reason": "attack detected" if flagged else "benign"
    }