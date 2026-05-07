# Extending PRADA for Sybil-Resilient Model Extraction Detection

PRADA (Juuti et al., EuroS&P 2019) detects model extraction attacks by checking whether an account's query-distance sequence follows a normal distribution. A Sybil attacker defeats this by splitting queries across N fake accounts, keeping each one under PRADA's 100-query warmup threshold. This project adds a second detection layer based on cross-account JS divergence. Sybil accounts all query with Jacobian-generated images that follow the same underlying pattern, so their dmin histograms converge. Benign accounts don't.The JS extension detects Sybil at N=4 through N=128 with zero false positives. Beyond N>=192, histogram sparsity collapses the JS separation gap, making detection increasingly unstable despite the detector still firing under the current threshold configuration.

## Setup

Python 3.10+. A CUDA GPU is not required but speeds up training significantly.

```bash
pip install -r requirements.txt
```

The victim model weights (`victim/victim_model.pt`) must exist before running any other component. The trained weights achieve 99.08% MNIST test accuracy.

If the weights file is missing, generate it using the training notebook:

victim/01_train_victim.ipynb

## How to Run

Run in this order. Each step depends on the previous. All commands from the project root.

**1. Start the victim API**
```bash
uvicorn api.server:app --port 8010
```
The API logs every query to `logs/queries.jsonl`. Leave it running during the attack.

**2. Run the Papernot JbDA attack**
```bash
python -m attacker.attack
```
Runs 6 rounds of Jacobian augmentation querying as `attacker_001`. Saves the substitute model to `attacker/substitute_model.pt`.

**3. Simulate a benign user**
```bash
python simulation/export.py
python simulation/benign.py
```
Exports 3000 MNIST seed images and queries the API as `benign_001`. Required for false-positive checks in evaluation.

**4. Run the full evaluation**
```bash
python evaluate.py
```
Runs PRADA and JS detection across N=1, 2, 5, 10, 64 Sybil scenarios and the benign baseline. Prints the detection summary table.

**5. Generate analysis result files**
```bash
python -m analysis.generate_results
```
Writes JSON results to `analysis/results/`. Already-existing files are skipped.

**6. Generate graphs**
```bash
python analysis/graphs.py
python analysis/roc_single.py
python analysis/roc_analysis.py
python analysis/prada_roc.py
python analysis/query_distribution_graph.py
```
Writes PNG files to `analysis/graphs/`.

## Project Structure

```
attacker/        JbDA attack: seed loading, victim querying, Jacobian augmentation, substitute training
api/             FastAPI victim server - logs every query with input vector and prediction to logs/queries.jsonl
defense/         PRADA (Algorithm 3 distances + Shapiro-Wilk detection) and JS divergence Sybil extension
simulation/      Experiment scripts: Sybil split sweep, mixed attacks, lambda sweep, query distributions
analysis/        Result generators, JS tradeoff tables, ROC analysis, graph scripts
cleanup/         Log trimming utility (trims to first N records per account)
utils/           Image load/save helpers
victim/          SimpleCNN model definition and trained weights
logs/            Query log (queries.jsonl) - created at runtime by the API
ui/              Simple browser-based visualization/debug interface
```

## Key Results

**Victim and attacker:**
- Victim model test accuracy: **99.08%**
- Substitute model agreement after 6 JbDA rounds: **96.56%**

**Single-account detection (PRADA baseline):**
- `attacker_001` W score: **0.9264** - flagged (DELTA = 0.96)
- `benign_001` W score: **0.9864** - clean, no false positive

**Sybil evasion and JS extension:**
- N=64 Sybil accounts: PRADA flags **0/64**. Each account accumulates ~90 dmin distances - below the MIN_QUERIES=100 threshold - because the first query per class seeds the reference set without generating a distance. N=64 is not arbitrary - it follows directly from 6400 total queries divided by MIN_QUERIES=100.
- JS extension detects N=64 Sybil with within-cluster mean JS = 0.1107, Sybil-benign mean JS = 0.2844. Zero false positives.
- Detection remains reliable through N=128. At N>=192, within-cluster JS begins exceeding cross-cluster JS as per-account histograms become sparse, indicating degradation of the underlying separation signal even though the detector still fires under the fixed threshold.

**ROC analysis:**
- JS detector pair-level AUC: **0.9978** (round-robin distribution, 64 Sybil vs 64 benign accounts)
- All three query distributions tested - round-robin, randomized, mixed 70-30 - all detected
- PRADA ROC AUC: **1.0** - but computed from only 2 W scores (1 attacker, 1 benign); not statistically informative

**Limitations:**
- JS detection becomes unstable at N>=192 as per-account histograms become sparse and the JS separation gap collapses or inverts
- Mixed 70-30 Sybil reduces the JS separation gap to 0.002 at some configurations (still detected, but margin is narrow)
- The JS threshold (0.15) was chosen on the same data used for evaluation - no held-out test set
- PRADA AUC=1.0 should not be cited as a robust result; it reflects 2-sample step-function behavior

## Reference

Juuti, M., Szyller, S., Marchal, S., & Asokan, N. (2019). PRADA: Protecting against DNN Model Stealing Attacks. *IEEE European Symposium on Security and Privacy (EuroS&P)*.
