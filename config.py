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
LAMBDA = 25.5 / 255    # FGSM perturbation step size; matches the paper's MNIST configuration

DELTA = 0.96           # W threshold from Juuti et al. MNIST experiments; flag if W < 0.96
MIN_QUERIES = 100      # Warmup: skip Shapiro-Wilk below this many dmin values. N=64 Sybil is derived from 6400 total queries / 100.

SYBIL_MIN_DMIN = 10
SYBIL_JS_THRESHOLD = 0.15   # Chosen empirically: within-Sybil mean JS = 0.1107, Sybil-benign mean JS = 0.2844; 0.15 sits cleanly between them
SYBIL_MIN_CLUSTER = 3
SYBIL_N_BINS = 50
