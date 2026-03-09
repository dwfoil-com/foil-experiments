# Foil Experiments

Experiments at the intersection of AI/ML and surf hydrofoils. Each directory is a self-contained project exploring a different aspect of foil design, physics, or rider biomechanics.

---

### [foil-rl-pump](foil-rl-pump/) - RL pump foil simulation
**Status:** mvp

Reinforcement learning experiments teaching an agent to pump foil. Includes a working Python RL approach (60+ second sustained flights) and an exploratory MuJoCo approach for more realistic physics simulation.

### [foil-mocap](foil-mocap/) - Motion capture from foil videos
**Status:** mvp

Extract body pose from pump foil videos using MediaPipe. Produces skeleton overlays, dot traces, and biomechanical features from any foil video.

### [hydrofoil](hydrofoil/) - Interactive surf hydrofoil explainer *(WIP)*
**Status:** in-progress

Long-form interactive article and simulation project inspired by [ciechanow.ski/airfoil](https://ciechanow.ski/airfoil/), but focused on surf hydrofoils with a rider on board. Covers the full rider-foil-board system: lift, drag, trim, stance, pumping, tails, shims, speed regimes, breach behavior, and structural loads. Built on `foilphysics` with surf-specific layers for rider stance, foot loading, and pump-cycle inputs.

### [foil-board-optimizer](foil-board-optimizer/) - Board structure topology optimization *(WIP)*
**Status:** in-progress

Autonomous topology optimization of hydrofoil board internal structures using FEA simulation and an AI-driven experimental outer loop (Karpathy autoresearch pattern). SIMP optimizer places material along the rider → deck → mast load path, replacing simple hollow shells with optimized internal ribs and lattices. Includes neural surrogate model and STL export for 3D printing.
