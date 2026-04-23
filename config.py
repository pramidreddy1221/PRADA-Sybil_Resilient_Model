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

# Sybil Detection
SYBIL_MIN_DMIN     = 10    # min dmin values per account to participate in cross-account analysis
SYBIL_JS_THRESHOLD = 0.20  # JS divergence below this → two accounts are "similar" (suspicious)
SYBIL_MIN_CLUSTER  = 3     # min accounts in a similar group before flagging as Sybil
SYBIL_N_BINS       = 50    # histogram bins for dmin distributions