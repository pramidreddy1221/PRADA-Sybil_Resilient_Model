import json
from pathlib import Path
from config import LOG_PATH

def load_logs(log_path: Path = LOG_PATH) -> list[dict]:
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records