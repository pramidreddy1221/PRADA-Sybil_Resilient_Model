import numpy as np
from collections import defaultdict

def compute_dmin_per_account(records: list[dict]) -> dict:
    by_account = defaultdict(list)
    for rec in records:
        by_account[rec["account_id"]].append(rec)

    results = {}

    for account_id, queries in by_account.items():
        Gc = defaultdict(list)
        Tc = defaultdict(float)
        DGc = defaultdict(list)
        D = []

        for query in queries:
            if "input_vector" not in query:
                continue

            x_vec = np.array(query["input_vector"], dtype=np.float32)
            c = query["pred"]

            if len(Gc[c]) == 0:
                Gc[c].append(x_vec)
                continue

            mat = np.stack(Gc[c], axis=0)
            dists = np.linalg.norm(mat - x_vec, axis=1)
            dmin = float(np.min(dists))

            D.append(dmin)

            if dmin > Tc[c]:
                DGc[c].append(dmin)
                Gc[c].append(x_vec)
                Tc[c] = max(Tc[c], np.mean(DGc[c]) - np.std(DGc[c]))

        results[account_id] = {
            "D": D,
            "n_queries": len(queries)
        }

    return results