#!/usr/bin/env python3
"""Modal GPU training for pump foil RL."""

import os
import sys
import modal

# Training config
FOIL_CONFIG = {
    'S_stab': 0.035,
    'stab_angle': -4.0,
    'S': 0.18,
    'AR': 8,
}
CHECKPOINT_PERCENTAGES = [1, 25, 50, 75, 100]

app = modal.App("pump-foil-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium==0.29.1",
        "stable-baselines3[extra]==2.1.0",
        "torch",
        "numpy",
    )
    .add_local_dir("foil_env", "/app/foil_env")
)


@app.function(image=image, gpu="A10G", timeout=7200)
def train_remote(timesteps, foil_config, percentages, resume_model_bytes=None):
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

    # If resuming, save the checkpoint locally first
    resume_path = None
    if resume_model_bytes:
        resume_path = "/tmp/resume_model.zip"
        with open(resume_path, "wb") as f:
            f.write(resume_model_bytes)
        print(f"Resuming from checkpoint ({len(resume_model_bytes)/1024:.0f} KB)")

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

    if resume_path:
        print(f"Loading model from checkpoint...")
        model = PPO.load(resume_path, env=env, device="cuda")
        # Update learning rate for continued training
        model.learning_rate = 3e-4
    else:
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


@app.local_entrypoint()
def main(timesteps: int = 1000000, output_dir: str = "checkpoints/run1", resume: str = None):
    print(f"=== PUMP FOIL TRAINING (Modal GPU) ===")
    print(f"Timesteps: {timesteps:,}")
    print(f"Output: {output_dir}/")
    if resume:
        print(f"Resuming from: {resume}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # Load resume checkpoint if provided
    resume_bytes = None
    if resume:
        with open(resume, "rb") as f:
            resume_bytes = f.read()
        print(f"Loaded checkpoint: {len(resume_bytes)/1024:.0f} KB")

    checkpoints = train_remote.remote(timesteps, FOIL_CONFIG, CHECKPOINT_PERCENTAGES, resume_bytes)

    for pct, data in checkpoints.items():
        path = f"{output_dir}/model_{pct}pct.zip"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")

    print(f"\nTraining complete! Checkpoints in {output_dir}/")
