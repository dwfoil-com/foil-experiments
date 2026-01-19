#!/usr/bin/env python3
"""Modal GPU training for pump foil with configurable rewards.

Supports:
- Resume from checkpoint
- Multiple reward modes: distance, velocity, pump, combined
- Configurable entropy for exploration
"""

import os
import modal

# 800 cm² foil - must pump to survive
FOIL_CONFIG = {
    'S': 0.08,
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

CHECKPOINT_PERCENTAGES = [1, 10, 25, 50, 75, 100]

app = modal.App("pump-foil-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "gymnasium==0.29.1",
        "stable-baselines3==2.1.0",
        "torch",
        "numpy",
        "tqdm",
        "rich",
        "matplotlib",
    )
    .add_local_dir("foil_env", "/app/foil_env")
)


@app.function(image=image, gpu="A10G", timeout=14400)
def train(
    timesteps: int,
    foil_config: dict,
    percentages: list,
    n_envs: int = 16,
    reward_mode: str = "distance",
    ent_coef: float = 0.01,
    resume_model_bytes: bytes = None,
    generate_videos: bool = True,
):
    import os
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from foil_env.pump_foil_env_survival import PumpFoilEnvSurvival, analyze_pumping

    if generate_videos:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from foil_env.foil_visualizer import draw_foil_and_rider, compute_body_positions, draw_motion_trails, draw_water

    save_path = "/tmp/checkpoints"
    video_path = "/tmp/videos"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # Resume handling
    resume_path = None
    if resume_model_bytes:
        resume_path = "/tmp/resume_model.zip"
        with open(resume_path, "wb") as f:
            f.write(resume_model_bytes)
        print(f"Resuming from checkpoint ({len(resume_model_bytes)/1024:.0f} KB)")

    def create_checkpoint_video(model, pct, foil_config, reward_mode, output_path, max_frames=500):
        """Generate video showing current model behavior."""
        env = PumpFoilEnvSurvival(config=foil_config, reward_mode=reward_mode)
        obs, _ = env.reset(seed=42)

        frames = []
        legs = []
        for step in range(max_frames):
            action, _ = model.predict(obs, deterministic=True)
            frames.append({
                't': step * env.dt, 'z': env.state.z, 'vx': env.state.vx,
                'theta': env.state.theta,
                'left_leg': env.left_leg_pos, 'right_leg': env.right_leg_pos,
                'left_arm': env.left_arm_pos, 'right_arm': env.right_arm_pos,
                'waist': env.waist_pos,
            })
            legs.append((env.left_leg_pos + env.right_leg_pos) / 2)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        duration = len(frames) * env.dt
        term_reason = info.get('termination_reason', 'max_steps')

        # Analyze pumping
        stats = analyze_pumping(legs[:200] if len(legs) >= 200 else legs)

        # Pre-compute body positions
        body_positions = []
        for frame in frames:
            pos = compute_body_positions(
                z=frame['z'], theta=frame['theta'],
                left_leg=frame['left_leg'], right_leg=frame['right_leg'],
                left_arm=frame['left_arm'], right_arm=frame['right_arm'],
                waist=frame['waist']
            )
            body_positions.append(pos)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_foil, ax_plots = axes

        frame_skip = 2
        render_frames = list(range(0, len(frames), frame_skip))

        def animate(frame_num):
            i = render_frames[frame_num]
            frame = frames[i]

            ax_foil.clear()
            draw_water(ax_foil, t=frame['t'], is_crashed=False, vx=frame['vx'])
            if i > 0:
                trail_start = max(0, i - 15)
                draw_motion_trails(ax_foil, body_positions[trail_start:i+1], current_vx=frame['vx'], dt=0.01)
            draw_foil_and_rider(
                ax_foil, z=frame['z'], theta=frame['theta'],
                left_leg=frame['left_leg'], right_leg=frame['right_leg'],
                left_arm=frame['left_arm'], right_arm=frame['right_arm'],
                waist=frame['waist'], t=frame['t'],
                foil_config=foil_config, is_crashed=False,
                vx=frame['vx'], draw_water_bg=False,
            )
            ax_foil.set_xlim(-1.5, 1.5)
            ax_foil.set_ylim(-1.2, 2.0)
            ax_foil.set_aspect('equal')
            ax_foil.set_facecolor('lightcyan')
            ax_foil.set_title(f"{pct}% | {frame['t']:.1f}s | {stats['freq']:.1f}Hz", fontsize=12, fontweight='bold')

            ax_plots.clear()
            t_arr = [f['t'] for f in frames[:i+1]]
            ax_plots.plot(t_arr, [f['z']*100 for f in frames[:i+1]], 'b-', label='Alt (cm)', lw=2)
            ax_plots.plot(t_arr, [f['vx']*10 for f in frames[:i+1]], 'g-', label='Vel x10', lw=2)
            ax_plots.plot(t_arr, [(f['left_leg']+f['right_leg'])/2*100 for f in frames[:i+1]], 'r-', label='Leg (cm)', lw=2)
            ax_plots.axhline(y=-50, color='red', ls='--', alpha=0.3)
            ax_plots.set_xlim(0, len(frames)*0.01)
            ax_plots.set_ylim(-60, 50)
            ax_plots.legend(loc='upper right', fontsize=8)
            ax_plots.grid(True, alpha=0.3)
            ax_plots.set_title(f"{duration:.1f}s | {term_reason}", fontsize=10)
            return []

        anim = FuncAnimation(fig, animate, frames=len(render_frames), interval=33, blit=False)
        FFMpegWriter(fps=30, bitrate=2000)
        anim.save(output_path, writer='ffmpeg', fps=30)
        plt.close()
        return duration, term_reason, stats

    class CheckpointCallback(BaseCallback):
        def __init__(self, total_timesteps, save_path, video_path, percentages, foil_config, reward_mode, gen_videos):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.save_path = save_path
            self.video_path = video_path
            self.percentages = percentages
            self.foil_config = foil_config
            self.reward_mode = reward_mode
            self.gen_videos = gen_videos
            self.saved = set()

        def _on_step(self):
            pct = (self.num_timesteps / self.total_timesteps) * 100
            for p in self.percentages:
                if p not in self.saved and pct >= p:
                    model_path = os.path.join(self.save_path, f"model_{p}pct")
                    self.model.save(model_path)
                    print(f"\n>>> Checkpoint: {p}%")
                    if self.gen_videos:
                        video_file = os.path.join(self.video_path, f"checkpoint_{p}pct.mp4")
                        try:
                            dur, reason, stats = create_checkpoint_video(
                                self.model, p, self.foil_config, self.reward_mode, video_file
                            )
                            print(f"    {dur:.1f}s | {stats['freq']:.1f}Hz | {reason}")
                        except Exception as e:
                            print(f"    Video failed: {e}")
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = PumpFoilEnvSurvival(config=foil_config, reward_mode=reward_mode)
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print(f"=== PUMP FOIL TRAINING ===")
    print(f"Foil: {foil_config['S']*10000:.0f} cm²")
    print(f"Reward: {reward_mode}")
    print(f"Entropy: {ent_coef}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Resume: {resume_path is not None}")
    print()

    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if resume_path:
        model = PPO.load(resume_path, env=env, device="cuda")
        model.ent_coef = ent_coef  # Allow changing entropy
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            verbose=1,
            device="cuda",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    callback = CheckpointCallback(
        timesteps, save_path, video_path, percentages, foil_config, reward_mode, generate_videos
    )
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    results = {'checkpoints': {}, 'videos': {}}
    for p in percentages:
        model_file = f"{save_path}/model_{p}pct.zip"
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                results['checkpoints'][p] = f.read()
        video_file = f"{video_path}/checkpoint_{p}pct.mp4"
        if os.path.exists(video_file):
            with open(video_file, "rb") as f:
                results['videos'][p] = f.read()
    return results


@app.local_entrypoint()
def main(
    timesteps: int = 2000000,
    output_dir: str = "checkpoints/exp1",
    reward_mode: str = "combined",
    ent_coef: float = 0.01,
    resume: str = None,
    no_video: bool = False,
):
    print(f"=== PUMP FOIL TRAINING ===")
    print(f"Foil: 800 cm²")
    print(f"Reward: {reward_mode}")
    print(f"Entropy: {ent_coef}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Resume: {resume or 'None'}")
    print(f"Output: {output_dir}/")
    print()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)

    resume_bytes = None
    if resume:
        with open(resume, "rb") as f:
            resume_bytes = f.read()
        print(f"Loaded checkpoint: {len(resume_bytes)/1024:.0f} KB")

    results = train.remote(
        timesteps, FOIL_CONFIG, CHECKPOINT_PERCENTAGES,
        n_envs=16,
        reward_mode=reward_mode,
        ent_coef=ent_coef,
        resume_model_bytes=resume_bytes,
        generate_videos=not no_video,
    )

    for pct, data in results['checkpoints'].items():
        path = f"{output_dir}/model_{pct}pct.zip"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")

    for pct, data in results['videos'].items():
        path = f"{output_dir}/videos/checkpoint_{pct}pct.mp4"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")

    print(f"\nDone! Output in {output_dir}/")
