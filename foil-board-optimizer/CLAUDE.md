# Foil Board Optimizer — Claude Code Instructions

## Outer Loop: Autonomous Experiment Runner

This project uses the [Karpathy autoresearch](https://github.com/karpathy/autoresearch) pattern. Claude Code acts as an autonomous researcher that runs experiments in a loop.

### How to start the outer loop

Open Claude Code in this directory and say:

> Read program.md and start optimizing. Run experiments autonomously.

### What the agent does each iteration

1. Read `program.md` for the research strategy
2. Read `results.tsv` to see what's been tried and the current best compliance
3. Form a hypothesis about what parameter change might improve stiffness
4. Modify the CONFIGURATION section of `optimize.py`
5. Run `python optimize.py` (~5 min per experiment)
6. Evaluate: if compliance improved (lower = stiffer = better), **keep** the change. If worse, **revert** `optimize.py`.
7. Commit kept improvements
8. Repeat — never stop, never ask for permission

### Key files

| File | Who modifies | Purpose |
|------|-------------|---------|
| `program.md` | Human | Research strategy and constraints |
| `optimize.py` | Agent | Experiment parameters (the knobs to turn) |
| `results.tsv` | Append-only | Experiment log — never delete rows |

### Rules

- Never modify `program.md` — only `optimize.py`
- Never delete `results.tsv` — only append
- Each experiment should complete in under 10 minutes
- Always commit after a kept improvement
- If something crashes, fix it and try again

### GPU acceleration via Modal

For high-resolution runs, use Modal cloud GPUs:

```bash
# One-time setup
pip install modal
modal setup  # creates account + authenticates

# Run on cloud
python modal_run.py --nelx 70 --nely 25 --nelz 10 --max-iter 200
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
