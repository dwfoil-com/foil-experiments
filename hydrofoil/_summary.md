# Hydrofoil

**Status:** `in-progress`

Interactive surf-hydrofoil explainer and simulation project, inspired by <https://ciechanow.ski/airfoil/> but focused on the full rider-foil-board system rather than airfoil sections alone. The goal is to explain trim, stance, pumping, tails, shims, speed regimes, breach behavior, and structural board loads using diagrams plus a simulation-backed model.

Initial plan is to reuse the force-model foundation from `foilphysics`, then add the pieces that matter for surf foils: rider foot placement, center-of-mass shifts, pumping inputs, near-surface behavior, and mast-track / bolt load calculations. The research scope is driven by common rider misconceptions from forum discussions, especially around trim, pumping efficiency, and why some setups feel "firm" or "mushy".

Stylistically, the benchmark is the teaching method of `airfoil`: interactive-first, scientifically careful, visually consistent, and built from simple experiments that make the invisible parts of the physics visible.
