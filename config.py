from pathlib import Path
import torch

# Paths
ROOT         = Path(__file__).resolve().parent
LOG_PATH     = ROOT / "logs" / "queries.jsonl"
MODEL_PATH   = ROOT / "victim" / "victim_model.pt"
MNIST_PATH   = ROOT / "victim" / "data"
SAVE_PATH    = ROOT / "attacker" / "substitute_model.pt"
RESULTS_PATH = ROOT / "attacker" / "attack_results.json"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Server
API_URL    = "http://127.0.0.1:8010/predict"

# Attacker
ROUNDS         = 6
SEED_PER_CLASS = 10
LAMBDA         = 25.5 / 255

# PRADA
DELTA       = 0.95
MIN_QUERIES = 100