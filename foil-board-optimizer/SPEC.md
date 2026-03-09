# Foil Board Internal Structure Optimizer — Specification

## Problem

Foil boards today are made hollow. When a rider stands on, pumps, or jumps on the board, energy is transmitted through the deck into the foil mast mount underneath. A hollow board absorbs too much of this energy — the board feels soft and compliant instead of stiff and responsive. Energy that should transfer directly from the rider's feet through the board into the foil is lost to flex and deformation.

## Objective

Design an optimal 3D internal structure (not hollow, not solid — an engineered hole/lattice pattern) for a foil board that:

- **Maximizes stiffness** — the board should feel rigid underfoot
- **Maximizes energy transfer** — force from the deck surface should transmit efficiently to the foil mast mount point
- **Minimizes material** — the board must remain light enough to be practical
- **Is 3D-printable** — the output structure must be manufacturable via additive manufacturing

## Inspiration

This is directly inspired by **Saab's autonomous aircraft structure design**, where topology optimization automatically discovers optimal internal structures that can be 3D printed. The same principle applies: let computation find structures that a human designer would never conceive.

## Key Physics

The board is loaded in multiple scenarios:

- **Riding (normal)** — steady-state weight on deck, force into mast mount
- **Pumping** — dynamic fore/aft weight transfer, cyclical loading
- **Jump landing** — high-impact loading across the deck
- **Carving** — asymmetric lateral loading

All load cases must route force efficiently from the deck surface down through the board thickness into the mast mount region. The optimization must work in full 3D — this is not a 2D cross-section problem.

## Approach

### Inner Loop: Topology Optimization

Use SIMP (Solid Isotropic Material with Penalization) topology optimization with 3D finite element analysis:

- 3D hexahedral mesh of the board volume
- One density variable per element (0 = void, 1 = solid)
- FEA solves for displacement under each load case
- Optimizer iterates to minimize compliance (maximize stiffness) subject to a volume fraction constraint
- Density filtering prevents checkerboard artifacts
- Output: 3D density field showing where material should exist

### Outer Loop: Autonomous Research Harness

This is the key architectural decision. The outer loop uses **Andrej Karpathy's autoresearch pattern** to drive long-running autonomous optimization:

- **`program.md`** — human-written research strategy (what to explore, what matters)
- **`optimize.py`** — agent-modifiable experiment configuration (parameters, approaches)
- **`results.tsv`** — append-only experiment log tracking what was tried and what worked

Claude Code reads the program, modifies experiment parameters, runs optimization, evaluates results, keeps or reverts changes, and loops autonomously. The outer loop has broad freedom to:

- Change mesh resolution, volume fractions, penalty factors
- Try different load case weightings
- Install new packages or try alternative optimization approaches
- Adjust convergence criteria
- Explore fundamentally different strategies for the objective

The outer loop follows three phases: **exploration** (broad parameter sweeps via Latin hypercube sampling) → **refinement** (narrowing around promising regions) → **validation** (confirming robustness of best designs).

### ML Acceleration (Future)

- **Surrogate model** — 3D CNN trained on optimization results to predict compliance without running full FEA
- **Neural topology predictor** — U-Net that directly predicts density fields, bypassing iterative SIMP

## Output

- Optimized 3D density field (`.npy`)
- 3D-printable STL file of the internal structure
- Convergence plots and cross-section visualizations
- Experiment log tracking all runs and parameter choices

## Success Criteria

1. The optimizer produces non-trivial internal structures (not just solid or hollow)
2. Optimized designs measurably outperform uniform-density baselines on stiffness
3. The autonomous outer loop discovers better configurations than manual parameter selection
4. Output STL files are valid and 3D-printable
