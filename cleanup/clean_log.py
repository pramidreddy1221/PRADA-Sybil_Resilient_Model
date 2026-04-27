import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_IN = ROOT / "logs" / "queries.jsonl"
LOG_OUT = ROOT / "logs" / "queries_clean.jsonl"

BENIGN_ACCOUNT = "benign_001"
BENIGN_LIMIT = 3000
DEFAULT_LIMIT = 6400


def limit_for(account_id: str) -> int:
    return BENIGN_LIMIT if account_id == BENIGN_ACCOUNT else DEFAULT_LIMIT


if __name__ == "__main__":
    if not LOG_IN.exists():
        print(f"ERROR: {LOG_IN} not found.")
        sys.exit(1)

    counts: dict[str, int] = {}
    kept: list[str] = []

    with LOG_IN.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = rec["account_id"]
            n = counts.get(aid, 0)
            if n < limit_for(aid):
                kept.append(line)
                counts[aid] = n + 1

    LOG_OUT.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(f"Input : {LOG_IN.relative_to(ROOT)}")
    print(f"Output: {LOG_OUT.relative_to(ROOT)}")
    print()
    print(f"  {'Account':<22} {'Kept':>6}  {'Limit':>6}")
    print("  " + "-" * 38)
    for aid, n in sorted(counts.items()):
        print(f"  {aid:<22} {n:>6}  {limit_for(aid):>6}")
    print(f"  {'TOTAL':<22} {sum(counts.values()):>6}")
    print()
    print("Original log unchanged.")
