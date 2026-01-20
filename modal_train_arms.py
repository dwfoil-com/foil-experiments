#!/usr/bin/env python3
"""
Modal GPU training with arms_full reward configuration.

Based on exploration results showing arms_full has best gradient (+0.17 sec/100k).
Uses all arm rewards combined: amplitude, speed, freq, sync (weight 2.0 each).
"""

import os
import modal

FOIL_CONFIG = {
    'S': 0.08,
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

# arms_full configuration (best from exploration)
ARMS_FULL_WEIGHTS = {
    'arm_amplitude_weight': 2.0,
    'arm_speed_weight': 2.0,
    'arm_freq_weight': 2.0,
    'arm_leg_sync_weight': 2.0,
    'jerk_weight': 0.0,
}

CHECKPOINT_PERCENTAGES = [10, 25, 50, 75, 100]

app = modal.App("pump-foil-arms-full")

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
def train_arms_full(
    timesteps: int,
    foil_config: dict,
    arm_weights: dict,
    percentages: list,
    n_envs: int = 16,
    ent_coef: float = 0.005,
):
    """Train with arms_full reward configuration."""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from foil_env.pump_foil_env_explore import PumpFoilEnvExplore

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from foil_env.foil_visualizer import draw_foil_and_rider, draw_water

    save_path = "/tmp/checkpoints"
    video_path = "/tmp/videos"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    def analyze_behavior(env, frames):
        """Analyze pumping behavior."""
        if len(frames) < 50:
            return {
                'arm_leg_corr': 0, 'pump_freq': 0, 'arm_range': 0, 'leg_range': 0,
                'velocity_loss': 0
            }

        leg_pos = np.array([(f['left_leg'] + f['right_leg']) / 2 for f in frames])
        arm_pos = np.array([(f['left_arm'] + f['right_arm']) / 2 for f in frames])
        velocities = np.array([f['vx'] for f in frames])

        if np.std(leg_pos) > 0.001 and np.std(arm_pos) > 0.001:
            corr = np.corrcoef(leg_pos, arm_pos)[0, 1]
        else:
            corr = 0

        leg_vel = np.diff(leg_pos) / env.dt
        crossings = np.where(np.diff(np.signbit(leg_vel)))[0]
        if len(crossings) >= 2:
            avg_half_period = np.mean(np.diff(crossings)) * env.dt
            freq = 0.5 / avg_half_period if avg_half_period > 0 else 0
        else:
            freq = 0

        velocity_loss = velocities[-1] - velocities[0] if len(velocities) > 1 else 0

        return {
            'arm_leg_corr': float(corr),
            'pump_freq': float(freq),
            'arm_range': float(np.degrees(arm_pos.max() - arm_pos.min())),
            'leg_range': float((leg_pos.max() - leg_pos.min()) * 100),
            'velocity_loss': float(velocity_loss),
        }

    def create_video(model, pct, output_path, max_frames=500):
        """Generate video showing model behavior."""
        env = PumpFoilEnvExplore(config=foil_config, **arm_weights)
        obs, _ = env.reset(seed=42)

        frames = []
        for step in range(max_frames):
            action, _ = model.predict(obs, deterministic=True)

            frames.append({
                't': step * env.dt, 'z': env.state.z, 'vx': env.state.vx,
                'theta': env.state.theta,
                'left_leg': env.left_leg_pos, 'right_leg': env.right_leg_pos,
                'left_arm': env.left_arm_pos, 'right_arm': env.right_arm_pos,
                'waist': env.waist_pos,
            })
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        duration = len(frames) * env.dt
        term_reason = info.get('termination_reason', 'max_steps')
        stats = analyze_behavior(env, frames)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_foil, ax_plots = axes

        frame_skip = 2
        render_frames = list(range(0, len(frames), frame_skip))

        def animate(frame_num):
            i = render_frames[frame_num]
            frame = frames[i]

            ax_foil.clear()
            draw_water(ax_foil, t=frame['t'], is_crashed=False, vx=frame['vx'])
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
            ax_foil.set_title(f"arms_full {pct}% | r={stats['arm_leg_corr']:.2f} | {stats['pump_freq']:.1f}Hz\n"
                             f"Leg: {stats['leg_range']:.0f}cm | Arm: {stats['arm_range']:.0f}deg",
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
            ax_plots.set_title(f"{duration:.1f}s | dV={stats['velocity_loss']:.2f}m/s | {term_reason}", fontsize=10)
            return []

        anim = FuncAnimation(fig, animate, frames=len(render_frames), interval=33, blit=False)
        anim.save(output_path, writer='ffmpeg', fps=30)
        plt.close()

        return {
            'duration': float(duration),
            'term_reason': str(term_reason),
            'pump_freq': float(stats['pump_freq']),
            'arm_leg_corr': float(stats['arm_leg_corr']),
            'arm_range': float(stats['arm_range']),
            'leg_range': float(stats['leg_range']),
            'velocity_loss': float(stats['velocity_loss']),
        }

    class CheckpointCallback(BaseCallback):
        def __init__(self, total_timesteps, save_path, video_path, percentages):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.save_path = save_path
            self.video_path = video_path
            self.percentages = percentages
            self.saved = set()
            self.stats = {}

        def _on_step(self):
            pct = (self.num_timesteps / self.total_timesteps) * 100
            for p in self.percentages:
                if p not in self.saved and pct >= p:
                    model_path = os.path.join(self.save_path, f"model_{p}pct")
                    self.model.save(model_path)
                    print(f"\n>>> Checkpoint: {p}%")
                    video_file = os.path.join(self.video_path, f"checkpoint_{p}pct.mp4")
                    try:
                        stats = create_video(self.model, p, video_file)
                        self.stats[p] = stats
                        print(f"    {stats['duration']:.1f}s | r={stats['arm_leg_corr']:.2f} | "
                              f"{stats['pump_freq']:.1f}Hz | arm={stats['arm_range']:.0f}deg | {stats['term_reason']}")
                    except Exception as e:
                        print(f"    Video failed: {e}")
                        import traceback
                        traceback.print_exc()
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = PumpFoilEnvExplore(config=foil_config, **arm_weights)
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print(f"=== ARMS_FULL TRAINING ===")
    print(f"Arm weights: {arm_weights}")
    print(f"Entropy: {ent_coef}")
    print(f"Timesteps: {timesteps:,}")
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
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    callback = CheckpointCallback(timesteps, save_path, video_path, percentages)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    results = {'checkpoints': {}, 'videos': {}, 'stats': callback.stats}
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
    output_dir: str = "checkpoints/arms_full_2M",
    ent_coef: float = 0.005,
):
    """Train arms_full model for 2M steps."""

    print(f"=== ARMS_FULL 2M TRAINING ===")
    print(f"Timesteps: {timesteps:,}")
    print(f"Entropy: {ent_coef}")
    print(f"Output: {output_dir}/")
    print()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)

    results = train_arms_full.remote(
        timesteps=timesteps,
        foil_config=FOIL_CONFIG,
        arm_weights=ARMS_FULL_WEIGHTS,
        percentages=CHECKPOINT_PERCENTAGES,
        n_envs=16,
        ent_coef=ent_coef,
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

    # Print summary
    print(f"\n{'='*80}")
    print("ARMS_FULL 2M TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<12} {'Duration':>10} {'Arm-Leg r':>12} {'Pump Hz':>10} {'Arm deg':>10} {'dV (m/s)':>12}")
    print("-"*80)

    stats = results.get('stats', {})
    for pct in sorted(stats.keys()):
        s = stats[pct]
        print(f"{pct}%{'':<10} {s['duration']:>10.1f}s {s['arm_leg_corr']:>12.2f} "
              f"{s['pump_freq']:>10.1f} {s['arm_range']:>10.0f} {s['velocity_loss']:>12.2f}")

    print(f"\nDone! Output in {output_dir}/")
