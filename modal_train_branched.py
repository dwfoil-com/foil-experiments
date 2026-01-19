#!/usr/bin/env python3
"""Modal GPU training with multiple branched experiments.

Runs parallel experiments with different configurations to test
the effect of increased velocity limits (ARM: 12 rad/s, LEG: 2 m/s).

Branches:
- baseline: Standard entropy (0.01)
- high_ent: High entropy (0.05) for more exploration
- very_high_ent: Very high entropy (0.1) for maximum exploration
- seed_42: Different seed for diversity
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

CHECKPOINT_PERCENTAGES = [10, 25, 50, 75, 100]

app = modal.App("pump-foil-branched")

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


@app.function(image=image, gpu="A10G", timeout=7200)
def train_branch(
    branch_name: str,
    timesteps: int,
    foil_config: dict,
    percentages: list,
    n_envs: int = 16,
    reward_mode: str = "combined",
    ent_coef: float = 0.01,
    seed: int = 42,
    generate_videos: bool = True,
):
    """Train a single branch experiment."""
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

    def create_checkpoint_video(model, pct, foil_config, reward_mode, output_path, max_frames=500):
        """Generate video showing current model behavior."""
        env = PumpFoilEnvSurvival(config=foil_config, reward_mode=reward_mode)
        obs, _ = env.reset(seed=seed)

        frames = []
        legs = []
        arms = []
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
            arms.append((env.left_arm_pos + env.right_arm_pos) / 2)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        duration = len(frames) * env.dt
        term_reason = info.get('termination_reason', 'max_steps')

        # Analyze pumping
        stats = analyze_pumping(legs[:200] if len(legs) >= 200 else legs)
        arm_range_deg = float(np.degrees(max(arms) - min(arms))) if arms else 0.0
        leg_range_cm = float((max(legs) - min(legs)) * 100) if legs else 0.0

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
            ax_foil.set_title(f"{branch_name} {pct}% | {frame['t']:.1f}s\nArm: {arm_range_deg:.0f}° Leg: {leg_range_cm:.0f}cm",
                             fontsize=11, fontweight='bold')

            ax_plots.clear()
            t_arr = [f['t'] for f in frames[:i+1]]
            ax_plots.plot(t_arr, [f['z']*100 for f in frames[:i+1]], 'b-', label='Alt (cm)', lw=2)
            ax_plots.plot(t_arr, [f['vx']*10 for f in frames[:i+1]], 'g-', label='Vel x10', lw=2)
            ax_plots.plot(t_arr, [(f['left_leg']+f['right_leg'])/2*100 for f in frames[:i+1]], 'r-', label='Leg (cm)', lw=2)
            ax_plots.plot(t_arr, [np.degrees(f['left_arm']+f['right_arm'])/2 for f in frames[:i+1]], 'm-', label='Arm (deg)', lw=1, alpha=0.7)
            ax_plots.axhline(y=-50, color='red', ls='--', alpha=0.3)
            ax_plots.set_xlim(0, len(frames)*0.01)
            ax_plots.set_ylim(-100, 100)
            ax_plots.legend(loc='upper right', fontsize=8)
            ax_plots.grid(True, alpha=0.3)
            ax_plots.set_title(f"{duration:.1f}s | {stats['freq']:.1f}Hz | {term_reason}", fontsize=10)
            return []

        anim = FuncAnimation(fig, animate, frames=len(render_frames), interval=33, blit=False)
        anim.save(output_path, writer='ffmpeg', fps=30)
        plt.close()
        # Return native Python types to avoid numpy serialization issues
        return {
            'duration': float(duration),
            'term_reason': str(term_reason),
            'pump_freq': float(stats['freq']),
            'arm_range_deg': float(arm_range_deg),
            'leg_range_cm': float(leg_range_cm),
        }

    class CheckpointCallback(BaseCallback):
        def __init__(self, total_timesteps, save_path, video_path, percentages, foil_config, reward_mode, gen_videos, branch_name):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.save_path = save_path
            self.video_path = video_path
            self.percentages = percentages
            self.foil_config = foil_config
            self.reward_mode = reward_mode
            self.gen_videos = gen_videos
            self.branch_name = branch_name
            self.saved = set()
            self.stats = {}

        def _on_step(self):
            pct = (self.num_timesteps / self.total_timesteps) * 100
            for p in self.percentages:
                if p not in self.saved and pct >= p:
                    model_path = os.path.join(self.save_path, f"model_{p}pct")
                    self.model.save(model_path)
                    print(f"\n>>> [{self.branch_name}] Checkpoint: {p}%")
                    if self.gen_videos:
                        video_file = os.path.join(self.video_path, f"checkpoint_{p}pct.mp4")
                        try:
                            stats = create_checkpoint_video(
                                self.model, p, self.foil_config, self.reward_mode, video_file
                            )
                            self.stats[p] = stats
                            print(f"    {stats['duration']:.1f}s | {stats['pump_freq']:.1f}Hz | Arm: {stats['arm_range_deg']:.0f}° | {stats['term_reason']}")
                        except Exception as e:
                            print(f"    Video failed: {e}")
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = PumpFoilEnvSurvival(config=foil_config, reward_mode=reward_mode)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    print(f"=== [{branch_name}] TRAINING ===")
    print(f"Seed: {seed}")
    print(f"Entropy: {ent_coef}")
    print(f"Timesteps: {timesteps:,}")
    print(f"MAX_LEG_VELOCITY: 2.0 m/s (was 1.0)")
    print(f"MAX_ARM_VELOCITY: 12.0 rad/s (was 3.0)")
    print()

    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

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
        seed=seed,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    callback = CheckpointCallback(
        timesteps, save_path, video_path, percentages, foil_config, reward_mode, generate_videos, branch_name
    )
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    results = {'checkpoints': {}, 'videos': {}, 'stats': callback.stats, 'branch': branch_name}
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
    timesteps: int = 500000,
    output_dir: str = "checkpoints/branched",
):
    """Run multiple branched experiments in parallel."""

    # Define branches with different configurations
    branches = [
        {"name": "ent_001", "ent_coef": 0.01, "seed": 42},    # Baseline entropy
        {"name": "ent_005", "ent_coef": 0.05, "seed": 42},    # Higher entropy
        {"name": "ent_010", "ent_coef": 0.10, "seed": 42},    # Very high entropy
        {"name": "seed_123", "ent_coef": 0.01, "seed": 123},  # Different seed
    ]

    print(f"=== BRANCHED TRAINING EXPERIMENT ===")
    print(f"Timesteps: {timesteps:,}")
    print(f"Output: {output_dir}/")
    print(f"Branches: {[b['name'] for b in branches]}")
    print(f"\nNew velocity limits:")
    print(f"  MAX_LEG_VELOCITY: 2.0 m/s (was 1.0)")
    print(f"  MAX_ARM_VELOCITY: 12.0 rad/s (was 3.0)")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # Launch all branches in parallel
    futures = []
    for branch in branches:
        branch_dir = f"{output_dir}/{branch['name']}"
        os.makedirs(branch_dir, exist_ok=True)
        os.makedirs(f"{branch_dir}/videos", exist_ok=True)

        future = train_branch.spawn(
            branch_name=branch['name'],
            timesteps=timesteps,
            foil_config=FOIL_CONFIG,
            percentages=CHECKPOINT_PERCENTAGES,
            n_envs=16,
            reward_mode="combined",
            ent_coef=branch['ent_coef'],
            seed=branch['seed'],
            generate_videos=True,
        )
        futures.append((branch['name'], future))
        print(f"Launched: {branch['name']} (ent={branch['ent_coef']}, seed={branch['seed']})")

    print(f"\nWaiting for {len(futures)} branches to complete...")
    print()

    # Collect results
    all_stats = {}
    for branch_name, future in futures:
        print(f"Collecting: {branch_name}...")
        results = future.get()

        branch_dir = f"{output_dir}/{branch_name}"

        for pct, data in results['checkpoints'].items():
            path = f"{branch_dir}/model_{pct}pct.zip"
            with open(path, "wb") as f:
                f.write(data)
            print(f"  Saved: {path}")

        for pct, data in results['videos'].items():
            path = f"{branch_dir}/videos/checkpoint_{pct}pct.mp4"
            with open(path, "wb") as f:
                f.write(data)

        all_stats[branch_name] = results.get('stats', {})

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Branch':<15} {'100% Duration':>12} {'Pump Hz':>10} {'Arm Range':>12} {'Leg Range':>12}")
    print("-"*70)

    for branch_name, stats in all_stats.items():
        if 100 in stats:
            s = stats[100]
            print(f"{branch_name:<15} {s['duration']:>10.1f}s {s['pump_freq']:>10.1f} {s['arm_range_deg']:>10.0f}° {s['leg_range_cm']:>10.0f}cm")
        else:
            print(f"{branch_name:<15} {'(no data)':<10}")

    print(f"\nDone! All results in {output_dir}/")
