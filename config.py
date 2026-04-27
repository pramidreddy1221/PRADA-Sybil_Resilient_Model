from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "queries.jsonl"
MODEL_PATH = ROOT / "victim" / "victim_model.pt"
MNIST_PATH = ROOT / "victim" / "data"
SAVE_PATH = ROOT / "attacker" / "substitute_model.pt"
RESULTS_PATH = ROOT / "attacker" / "attack_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

API_URL = "http://127.0.0.1:8010/predict"

ROUNDS = 6
SEED_PER_CLASS = 10
LAMBDA = 25.5 / 255

DELTA = 0.96
MIN_QUERIES = 100

SYBIL_MIN_DMIN = 10
SYBIL_JS_THRESHOLD = 0.15
SYBIL_MIN_CLUSTER = 3
SYBIL_N_BINS = 50
