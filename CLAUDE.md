# Pump Foil RL Project

## Overview

Training an RL agent to pump foil - maintaining flight through coordinated body movements (legs, arms, waist) that generate vertical forces and pitch torque on a hydrofoil.

## Project Structure

```
foilpump/
├── train.py           # Standard training script
├── create_video.py    # Standard video creation
├── foil_env/          # Core environment
│   ├── foil_physics.py           # Hydrodynamic physics
│   ├── body_model.py             # Rider body mechanics
│   ├── foil_visualizer.py        # Reusable visualization
│   └── pump_foil_env_curriculum.py  # Main RL environment
├── checkpoints/       # Saved models
├── requirements.txt
├── venv/
└── archive/           # Old experiments (reference only)
```

## Standard Training Procedure

### 1. Training Requirements

Every training run MUST:
- Save checkpoints at **1%, 25%, 50%, 75%, 100%** of training
- Use consistent foil config (defined in train.py)
- Output to a named checkpoint directory

### 2. Running Training

```bash
# Local (slower, ~500k steps)
python train.py --timesteps 500000 --output checkpoints/experiment_name

# Modal GPU (faster, recommended for >500k steps)
python train.py --modal --timesteps 1000000 --output checkpoints/experiment_name
```

### 3. After Training: Create Video

Every training run should produce an evolution video:

```bash
python create_video.py --checkpoint-dir checkpoints/experiment_name -o experiment_evolution.mp4
```

## Objective & Constraints

### What We're Trying to Achieve
- Agent learns to **pump** (rhythmic leg motion at ~2Hz, full range)
- Maintains flight as long as possible (target: >10s)
- Maintains velocity (target: ~4.5 m/s)

### Physical Constraints
- **Ceiling**: Foil breaches at z ≥ 0.2m (crash)
- **Floor**: Board touchdown at z ≤ -0.5m (crash)
- **Energy budget**: 6000J max (prevents unrealistic sustained output)
- **Stall velocity**: < 1.5 m/s (crash)

### Success Metrics (per episode)
| Metric | Poor | Good | Excellent |
|--------|------|------|-----------|
| Flight duration | <3s | 5-8s | >10s |
| Leg range | <30% | 50-80% | >90% |
| Pump frequency | <1Hz | 1.5-2Hz | ~2Hz |

## Video Format

Standard 4-panel comparison showing training evolution:
- **Top row**: 4 foil visualizations (1%, 25%, 75%, 100% checkpoints)
- **Bottom row**: 5 time series overlaid (altitude, velocity, leg, arm, waist)
- Shows how pumping behavior emerges through training

## Foil Configuration

Current "training foil" (more stable for learning):
```python
FOIL_CONFIG = {
    'S_stab': 0.035,     # Stabilizer area (m²) - larger = stable
    'stab_angle': -4.0,  # Stabilizer angle (deg)
    'S': 0.18,           # Wing area (m²)
    'AR': 8,             # Aspect ratio - lower = stable
}
```

## Key Insights

1. **Curriculum learning works**: Forcing pumping initially, then releasing control
2. **Frequency reward helps**: Rewarding phase alignment with 2Hz reference
3. **Deweighting matters**: Arm and torso acceleration creates vertical reaction forces
4. **RL finds nuanced strategies**: May not match "optimal manual" exactly but finds what works

See `docs/pump-physics.md` for detailed physics analysis and energy tradeoffs.

## Do NOT

- Create analysis images unless explicitly requested
- Use old environments from archive/
- Skip checkpoint saving during training
- Skip video creation after training
- Add new features without updating this doc
