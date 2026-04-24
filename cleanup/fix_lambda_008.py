"""
cleanup/fix_lambda_008.py — Trim lambda_008 records to first 6400.

Loads logs/queries.jsonl, keeps only the first 6400 records for
account_id "lambda_008", leaves all other accounts untouched,
and writes the result back in place.

Run with:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python cleanup/fix_lambda_008.py
"""

import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import LOG_PATH

ACCOUNT   = "lambda_008"
KEEP      = 6400

records = []
with LOG_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

before_total  = len(records)
before_target = sum(1 for r in records if r["account_id"] == ACCOUNT)

seen = 0
cleaned = []
for r in records:
    if r["account_id"] == ACCOUNT:
        if seen < KEEP:
            cleaned.append(r)
            seen += 1
    else:
        cleaned.append(r)

after_total  = len(cleaned)
after_target = sum(1 for r in cleaned if r["account_id"] == ACCOUNT)

with LOG_PATH.open("w", encoding="utf-8") as f:
    for r in cleaned:
        f.write(json.dumps(r) + "\n")

print(f"Account : {ACCOUNT}")
print(f"Before  : {before_target} records  (total log: {before_total})")
print(f"After   : {after_target} records  (total log: {after_total})")
print(f"Removed : {before_target - after_target} records")
print(f"Written : {LOG_PATH}")
