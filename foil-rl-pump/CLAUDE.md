# Foil RL Pump — Claude Code Instructions

## Quick Start

```bash
cd python-rl
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Standard Workflow

```bash
# Train a new model (saves checkpoints at 1%, 25%, 50%, 75%, 100%)
python train.py --timesteps 500000 --output checkpoints/my_run

# Create evolution video from checkpoints
python create_video.py --checkpoint-dir checkpoints/my_run -o evolution.mp4
```

## Outer Loop: Autonomous Training Experiments

The agent can run training experiments autonomously — varying reward functions, physics parameters, curriculum schedules, and foil configurations to discover better pumping strategies.

### Key files to modify

| File | Purpose |
|------|---------|
| `foil_env/config.py` | Foil physics parameters, foil presets |
| `foil_env/pump_foil_env_curriculum.py` | Reward function, curriculum schedule |
| `foil_env/foil_physics.py` | Hydrodynamic model |
| `train.py` | Training hyperparameters |

### What to try

- Different foil sizes (wing area, stabilizer area)
- Reward function weights (altitude vs velocity vs frequency vs efficiency)
- Curriculum pacing (how fast to release control)
- Physics parameters (drag coefficients, pump efficiency)
- PPO hyperparameters (learning rate, batch size, n_steps)

### Evaluate by

- Flight duration (longer = better)
- Pump frequency (should be near 2-2.5 Hz for realism)
- Energy efficiency (sustained flight with minimal effort)
- Watch the evolution video to see if the technique looks realistic
