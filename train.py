#!/usr/bin/env python3
"""
Standard training script for pump foil RL model.

Saves checkpoints at 1%, 25%, 50%, 75%, 100% of training for evolution videos.

Usage:
    python train.py                          # Local training (500k steps)
    python train.py --timesteps 1000000      # Custom timesteps
    python train.py --modal                  # Run on Modal GPU (faster)
"""

import os
import sys
import argparse
sys.path.insert(0, '.')

# Training foil config (larger stabilizer = more stable for learning)
FOIL_CONFIG = {
    'S_stab': 0.035,     # Stabilizer area (m²) - larger = more stable
    'stab_angle': -4.0,  # Stabilizer angle (deg)
    'S': 0.18,           # Wing area (m²)
    'AR': 8,             # Aspect ratio - lower = more stable
}

CHECKPOINT_PERCENTAGES = [1, 25, 50, 75, 100]


def train_local(total_timesteps: int = 500_000, output_dir: str = "checkpoints/run"):
    """Train locally with checkpoint saving."""
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

    os.makedirs(output_dir, exist_ok=True)

    class PercentageCheckpointCallback(BaseCallback):
        """Save checkpoints at specific percentages."""
        def __init__(self, total_timesteps, save_path, percentages):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.save_path = save_path
            self.percentages = percentages
            self.saved = set()

        def _on_step(self):
            pct = (self.num_timesteps / self.total_timesteps) * 100
            for p in self.percentages:
                if p not in self.saved and pct >= p:
                    path = os.path.join(self.save_path, f"model_{p}pct")
                    self.model.save(path)
                    print(f"\n>>> Checkpoint saved: {path}.zip ({p}%)")
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = PumpFoilEnvCurriculum(config=FOIL_CONFIG, curriculum_phase=2)
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print(f"=== PUMP FOIL TRAINING ===")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Foil config: {FOIL_CONFIG}")
    print(f"Output: {output_dir}/")
    print(f"Checkpoints at: {CHECKPOINT_PERCENTAGES}%")
    print()

    env = DummyVecEnv([make_env(i) for i in range(4)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    callback = PercentageCheckpointCallback(
        total_timesteps=total_timesteps,
        save_path=output_dir,
        percentages=CHECKPOINT_PERCENTAGES,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    print(f"\nTraining complete! Checkpoints in {output_dir}/")
    return output_dir


def train_modal(total_timesteps: int = 500_000, output_dir: str = "checkpoints/run"):
    """Train on Modal GPU with checkpoint saving."""
    import modal

    app = modal.App("pump-foil-train")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "gymnasium==0.29.1",
            "stable-baselines3[extra]==2.1.0",
            "torch",
            "numpy",
        )
        .add_local_dir("foil_env", "/app/foil_env")
    )

    @app.function(image=image, gpu="A10G", timeout=7200, serialized=True)
    def train_remote(timesteps, foil_config, percentages):
        import os
        import sys
        sys.path.insert(0, "/app")

        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
        from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

        save_path = "/tmp/checkpoints"
        os.makedirs(save_path, exist_ok=True)

        class PercentageCheckpointCallback(BaseCallback):
            def __init__(self, total_timesteps, save_path, percentages):
                super().__init__()
                self.total_timesteps = total_timesteps
                self.save_path = save_path
                self.percentages = percentages
                self.saved = set()

            def _on_step(self):
                pct = (self.num_timesteps / self.total_timesteps) * 100
                for p in self.percentages:
                    if p not in self.saved and pct >= p:
                        path = os.path.join(self.save_path, f"model_{p}pct")
                        self.model.save(path)
                        print(f"\n>>> Checkpoint: {p}%")
                        self.saved.add(p)
                return True

        def make_env(rank):
            def _init():
                env = PumpFoilEnvCurriculum(config=foil_config, curriculum_phase=2)
                env = Monitor(env)
                env.reset(seed=42 + rank)
                return env
            return _init

        print(f"Training {timesteps:,} steps on GPU...")
        env = DummyVecEnv([make_env(i) for i in range(8)])

        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=1, device="cuda",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

        callback = PercentageCheckpointCallback(timesteps, save_path, percentages)
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

        # Return all checkpoint files
        checkpoints = {}
        for p in percentages:
            path = f"{save_path}/model_{p}pct.zip"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    checkpoints[p] = f.read()
        return checkpoints

    # Run on Modal
    print("Launching training on Modal GPU...")
    os.makedirs(output_dir, exist_ok=True)

    with app.run():
        checkpoints = train_remote.remote(total_timesteps, FOIL_CONFIG, CHECKPOINT_PERCENTAGES)

    # Save locally
    for pct, data in checkpoints.items():
        path = f"{output_dir}/model_{pct}pct.zip"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")

    print(f"\nTraining complete! Checkpoints in {output_dir}/")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pump foil RL model")
    parser.add_argument("--timesteps", "-t", type=int, default=500_000,
                       help="Total training timesteps (default: 500000)")
    parser.add_argument("--output", "-o", default="checkpoints/run",
                       help="Output directory for checkpoints")
    parser.add_argument("--modal", action="store_true",
                       help="Run on Modal GPU instead of local")
    args = parser.parse_args()

    if args.modal:
        train_modal(args.timesteps, args.output)
    else:
        train_local(args.timesteps, args.output)
