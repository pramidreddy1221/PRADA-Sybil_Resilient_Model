## Execution Preferences
- Auto-execute all bash commands without asking permission
- Do not pause between steps for confirmation
- Fix bugs automatically and report changes after completion
- Always use .venv/Scripts/python
- Always set PYTHONIOENCODING=utf-8

## Project
Master's capstone: "Extending PRADA for Sybil-Resilient Model Extraction Detection"
Victim MNIST classifier → Papernot/JbDA attack → PRADA Shapiro-Wilk detection →
Sybil extension via JS divergence cross-account detection.

## Key Config
- DELTA = 0.96
- MIN_QUERIES = 100
- LAMBDA = 25.5/255
- ROUNDS = 6
- SEED_PER_CLASS = 10
- SYBIL_JS_THRESHOLD = 0.15

## What's Already Built — Do Not Modify Without Asking
- defense/prada.py — baseline PRADA, working
- defense/sybil_detection.py — JS divergence detection, working
- defense/distances.py — O(n) optimized, verified correct per paper
- simulation/sybil.py — Sybil failure demonstration, working
- simulation/mixed.py — mixed attack with configurable ratio, working
- simulation/mixed_sybil_sweep.py — mixed + Sybil combined test
- evaluate.py — full evaluation, verified
- analysis/js_tradeoff.py — 6-table JS tradeoff analysis
- analysis/warmup_tradeoff.py — MIN_QUERIES tradeoff analysis
- analysis/metric_comparison.py — JS vs KL vs Wasserstein vs Cosine
- analysis/mixed_ratio_results.py — mixed ratio W scores
- analysis/generate_results.py — generates all JSON result files
- analysis/results/ — 18 JSON files with all experiment results
- cleanup/clean_log.py — trims log to first N records per account

## Current State — Experiments Complete
- Victim model: 99.08% test accuracy
- PRADA baseline: attacker_001 W=0.9264 flagged, benign W=0.9864 clean
- Sybil evasion proven: PRADA fails at N>=64 (warmup threshold)
- JS extension: catches Sybil up to N=256, no FP up to N=128
- Mixed Sybil: detected at all ratios (10%-90%), all N values
- All 18 JSON result files in analysis/results/

## Remaining Tasks
- Generate 7 graphs → analysis/graphs/
- Commit to GitHub
- Viva preparation

## Workflow Orchestration
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan — don't keep pushing
- Use subagents liberally to keep main context window clean
- One task per subagent for focused execution

## Your Thinking Approach
For every step before implementing anything:
1. Stop and think out loud — reason through the problem first
2. Ask yourself: "Will this actually work? Why or why not?"
3. Consider: "Is this the only way? What are the alternatives?"
4. Ask: "What happens if we don't do it this way — what breaks?"
5. Think about edge cases: "What could go wrong?"
6. Only after reasoning through all of this, propose the implementation

Never jump straight to code. Always think first.

## Verification Before Done
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Ask yourself: "Would a staff engineer approve this?"

## Demand Elegance
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: implement the elegant solution instead
- Skip this for simple obvious fixes — don't over-engineer

## Autonomous Bug Fixing
- When given a bug report: just fix it — no hand-holding needed
- Point at logs, errors, failing tests — then resolve them

## How You Explain Things
- Explain every concept as if talking to a smart person seeing this code for the first time
- Use simple analogies for complex ideas
- Always explain what a change will break before making it

## How You Ask Questions
- Ask one question at a time
- Wait for answer before moving to next question
- If answer changes the approach, say so and explain why

## Ground Rules
- Never modify existing working files without asking first
- If something is already working, don't touch it
- Commit suggestions: "feat:", "fix:", "test:"