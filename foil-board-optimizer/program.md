# Foil Board Internal Structure Optimization — Agent Research Program

You are an autonomous research agent optimizing the internal 3D structure of a
hydrofoil board. Your goal is to **minimize compliance** (maximize stiffness)
while keeping volume fraction at or below the target, so that energy from the
rider transfers efficiently through the deck into the foil mast mount rather
than being absorbed by board flex.

## Your Mission

Find the optimal combination of topology optimization parameters that produces
the stiffest possible 3D-printable internal board structure. You do this by
modifying `optimize.py`, running it, evaluating the result, and iterating.

## How This Works

1. **Read** this file and `optimize.py` to understand the current state
2. **Read** `results.tsv` to see all past experiments and their outcomes
3. **Form a hypothesis** about what change might improve stiffness
4. **Modify** `optimize.py` — change parameters, try different approaches
5. **Run** `cd /home/user/ai-research/projects/foil-board-optimizer && python optimize.py`
6. The script prints a result line and appends it to `results.tsv`
7. **Evaluate**: if the new compliance is lower (stiffer), KEEP the change.
   If worse, REVERT `optimize.py` to the previous version.
8. **Commit** your changes with a descriptive message
9. **Repeat** — never stop, never ask for permission. The human might be asleep.

## What You Can Change in optimize.py

### Mesh Resolution (exploration → refinement)
- `nelx`, `nely`, `nelz` — start coarse (14x5x2), go finer (56x20x8)
- Higher resolution = more accurate but slower. Budget ~5 min per run.
- Strategy: explore at coarse resolution, refine best configs at high resolution.

### SIMP Parameters
- `volfrac` (0.15–0.60): target material fraction. Lower = lighter but weaker.
- `penal` (2.0–5.0): penalization power. Higher pushes densities toward 0/1.
- `rmin` (1.0–3.0): filter radius. Controls minimum feature size.
- `max_iter` (50–300): optimization iterations.
- `use_heaviside` (True/False): sharper boundaries, more 3D-printable.
- `move_limit` (0.05–0.3): how much density can change per iteration.

### Board Geometry
- `board_length` (1.2–1.6 m)
- `board_width` (0.40–0.55 m)
- `board_thickness` (0.06–0.14 m)
- Mast mount position and size

### Material Properties
- `E0`: Young's modulus (1e9 for PLA, 2e9 for PETG, 3.5e9 for carbon-filled)
- `nu`: Poisson's ratio (0.30–0.40)

### Load Cases
- Select which load cases to optimize against (riding, pumping, jumping, carving)
- Weight rider mass (60–100 kg)

### Advanced Ideas to Try
- **Continuation method**: start with low penalization (p=1), gradually increase
- **Multi-resolution**: optimize coarse, interpolate to fine, refine
- **Asymmetric loading**: weight certain load cases more than others
- **Anisotropic filtering**: different filter radii in X/Y/Z
- **Custom passive regions**: force ribs at specific locations
- **Two-phase optimization**: first optimize for riding, then tune for jumps

## Success Metric

The primary metric is **compliance** (lower = stiffer = better). Secondary
metrics: volume fraction (should be near target), max displacement, convergence.

Check `results.tsv` for the current best. Beat it.

## Rules

- Each experiment should run in under 10 minutes. If it's too slow, reduce mesh.
- Always log results to `results.tsv` (the script does this automatically).
- Never modify `program.md` — only modify `optimize.py`.
- Never delete `results.tsv`.
- Always commit after a kept improvement.
- If something crashes, fix it and try again. Don't give up.
- Think creatively. The parameter space is large. Use insights from past results.

## Current Best

Check `results.tsv` for the latest. Your job is to beat the best compliance
while keeping volume fraction at or below the target.
