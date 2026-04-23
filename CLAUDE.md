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
- DELTA = 0.95
- MIN_QUERIES = 100
- LAMBDA = 25.5/255
- ROUNDS = 6
- SEED_PER_CLASS = 10
- SYBIL_JS_THRESHOLD = 0.20

## What's Already Built — Do Not Modify Without Asking
- defense/prada.py — baseline PRADA, working
- defense/sybil_detection.py — JS divergence detection, working
- simulation/sybil.py — Sybil failure demonstration, working
- simulation/mixed.py — mixed attack with configurable ratio, working
- evaluate.py — full evaluation, verified

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