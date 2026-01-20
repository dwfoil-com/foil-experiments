#!/usr/bin/env python3
"""
Modal GPU exploration script.

Fine-tunes from baseline checkpoint with different reward configurations.
Runs short experiments (200k steps) to find which features show steepest gradient.
"""

import os
import modal

FOIL_CONFIG = {
    'S': 0.08,
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

# Experiment configurations to test
# Each config tests a different reward feature or combination
EXPERIMENTS = [
    # Baseline (no extra features)
    {"name": "baseline", "weights": {}},

    # Individual features
    {"name": "arm_amplitude", "weights": {"arm_amplitude_weight": 3.0}},
    {"name": "arm_speed", "weights": {"arm_speed_weight": 3.0}},
    {"name": "arm_freq", "weights": {"arm_freq_weight": 3.0}},
    {"name": "arm_leg_sync", "weights": {"arm_leg_sync_weight": 3.0}},
    {"name": "jerk_penalty", "weights": {"jerk_weight": 0.3}},

    # Combinations (arms focus)
    {"name": "arms_full", "weights": {
        "arm_amplitude_weight": 2.0,
        "arm_speed_weight": 2.0,
        "arm_freq_weight": 2.0,
        "arm_leg_sync_weight": 2.0,
    }},

    # Arms + jerk (smooth coordinated arm motion)
    {"name": "arms_smooth", "weights": {
        "arm_amplitude_weight": 2.0,
        "arm_freq_weight": 2.0,
        "arm_leg_sync_weight": 3.0,
        "jerk_weight": 0.2,
    }},
]

app = modal.App("pump-foil-explore")

# Volume for baseline checkpoint
baseline_volume = modal.Volume.from_name("pump-foil-checkpoints", create_if_missing=True)

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


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/data": baseline_volume})
def run_experiment(
    experiment_name: str,
    reward_weights: dict,
    baseline_path: str,
    timesteps: int,
    foil_config: dict,
):
    """Run a single exploration experiment."""
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

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Reward weights: {reward_weights}")
    print(f"{'='*60}\n")

    n_envs = 8

    def analyze_behavior(env, frames):
        """Analyze pumping behavior."""
        if len(frames) < 50:
            return {'arm_leg_corr': 0, 'pump_freq': 0, 'arm_range': 0, 'leg_range': 0, 'velocity_loss': 0}

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

    def evaluate_model(model, max_frames=1000):
        """Evaluate model and return metrics."""
        env = PumpFoilEnvExplore(config=foil_config, **reward_weights)
        obs, _ = env.reset(seed=42)

        frames = []
        total_reward = 0
        for step in range(max_frames):
            action, _ = model.predict(obs, deterministic=True)

            frames.append({
                't': step * env.dt, 'z': env.state.z, 'vx': env.state.vx,
                'theta': env.state.theta,
                'left_leg': env.left_leg_pos, 'right_leg': env.right_leg_pos,
                'left_arm': env.left_arm_pos, 'right_arm': env.right_arm_pos,
                'waist': env.waist_pos,
            })
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        duration = len(frames) * env.dt
        term_reason = info.get('termination_reason', 'max_steps')
        stats = analyze_behavior(env, frames)

        return {
            'duration': duration,
            'term_reason': term_reason,
            'total_reward': total_reward,
            **stats
        }

    def create_video(model, output_path, max_frames=500):
        """Generate video showing model behavior."""
        env = PumpFoilEnvExplore(config=foil_config, **reward_weights)
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
            ax_foil.set_title(f"{experiment_name} | r={stats['arm_leg_corr']:.2f} | {stats['pump_freq']:.1f}Hz\n"
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

        return stats

    class ProgressCallback(BaseCallback):
        def __init__(self, total_timesteps, eval_interval):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.eval_interval = eval_interval
            self.metrics = []
            self.last_eval = 0

        def _on_step(self):
            if self.num_timesteps - self.last_eval >= self.eval_interval:
                metrics = evaluate_model(self.model, max_frames=500)
                metrics['timesteps'] = self.num_timesteps
                self.metrics.append(metrics)
                print(f"  [{self.num_timesteps:,}] {metrics['duration']:.1f}s | r={metrics['arm_leg_corr']:.2f} | "
                      f"{metrics['pump_freq']:.1f}Hz | arm={metrics['arm_range']:.0f}deg | {metrics['term_reason']}")
                self.last_eval = self.num_timesteps
            return True

    def make_env(rank):
        def _init():
            env = PumpFoilEnvExplore(config=foil_config, **reward_weights)
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    # Create environment (no VecNormalize - baseline was trained without it)
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Load baseline model
    print(f"Loading baseline from {baseline_path}...")
    model = PPO.load(baseline_path, env=env, device="cuda")

    # Initial evaluation
    print("\nInitial (baseline) evaluation:")
    initial_metrics = evaluate_model(model, max_frames=1000)
    print(f"  {initial_metrics['duration']:.1f}s | r={initial_metrics['arm_leg_corr']:.2f} | "
          f"{initial_metrics['pump_freq']:.1f}Hz | arm={initial_metrics['arm_range']:.0f}deg")

    # Fine-tune
    print(f"\nFine-tuning for {timesteps:,} steps...")
    callback = ProgressCallback(timesteps, eval_interval=50000)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True, reset_num_timesteps=False)

    # Final evaluation
    print("\nFinal evaluation:")
    final_metrics = evaluate_model(model, max_frames=1000)
    print(f"  {final_metrics['duration']:.1f}s | r={final_metrics['arm_leg_corr']:.2f} | "
          f"{final_metrics['pump_freq']:.1f}Hz | arm={final_metrics['arm_range']:.0f}deg")

    # Skip video generation for exploration (faster)
    video_data = None

    # Compute gradients (improvement per 100k steps)
    duration_gradient = (final_metrics['duration'] - initial_metrics['duration']) / (timesteps / 100000)
    arm_range_gradient = (final_metrics['arm_range'] - initial_metrics['arm_range']) / (timesteps / 100000)
    corr_gradient = (final_metrics['arm_leg_corr'] - initial_metrics['arm_leg_corr']) / (timesteps / 100000)

    # Convert numpy types to Python types for serialization
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return obj

    return to_python({
        'name': experiment_name,
        'weights': reward_weights,
        'initial': initial_metrics,
        'final': final_metrics,
        'gradients': {
            'duration': duration_gradient,
            'arm_range': arm_range_gradient,
            'arm_leg_corr': corr_gradient,
        },
        'progress': callback.metrics,
    })


@app.function(image=image, timeout=600, volumes={"/data": baseline_volume})
def upload_baseline(model_data: bytes, path: str):
    """Upload baseline checkpoint to volume."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(model_data)
    print(f"Uploaded baseline to {path}")


@app.local_entrypoint()
def main(
    timesteps: int = 200000,
    output_dir: str = "checkpoints/explore",
    baseline: str = "checkpoints/ent005_2M/model_100pct.zip",
):
    """Run exploration experiments."""

    print(f"=== EXPLORATION MODE ===")
    print(f"Baseline: {baseline}")
    print(f"Timesteps per experiment: {timesteps:,}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)

    # Upload baseline to volume
    print("Uploading baseline checkpoint...")
    with open(baseline, "rb") as f:
        baseline_data = f.read()
    upload_baseline.remote(baseline_data, "/data/baseline.zip")

    # Run experiments in parallel
    print(f"\nLaunching {len(EXPERIMENTS)} experiments in parallel...")
    futures = []
    for exp in EXPERIMENTS:
        future = run_experiment.spawn(
            experiment_name=exp["name"],
            reward_weights=exp["weights"],
            baseline_path="/data/baseline.zip",
            timesteps=timesteps,
            foil_config=FOIL_CONFIG,
        )
        futures.append((exp["name"], future))

    # Collect results
    results = []
    for name, future in futures:
        print(f"\nWaiting for {name}...")
        result = future.get()
        results.append(result)
        print(f"  Got results for {name}")

    # Print summary
    print(f"\n{'='*100}")
    print("EXPLORATION RESULTS")
    print(f"{'='*100}")
    print(f"{'Experiment':<20} {'Initial':>10} {'Final':>10} {'Gradient':>12} {'Arm r':>8} {'Arm deg':>10} {'Freq':>8}")
    print(f"{'':20} {'(sec)':>10} {'(sec)':>10} {'(sec/100k)':>12} {'final':>8} {'final':>10} {'final':>8}")
    print("-"*100)

    # Sort by duration gradient
    results.sort(key=lambda x: x['gradients']['duration'], reverse=True)

    for r in results:
        print(f"{r['name']:<20} "
              f"{r['initial']['duration']:>10.1f} "
              f"{r['final']['duration']:>10.1f} "
              f"{r['gradients']['duration']:>+12.2f} "
              f"{r['final']['arm_leg_corr']:>+8.2f} "
              f"{r['final']['arm_range']:>10.0f} "
              f"{r['final']['pump_freq']:>8.1f}")

    # Identify best
    best = results[0]
    print(f"\n{'='*100}")
    print(f"BEST: {best['name']} with gradient {best['gradients']['duration']:+.2f} sec/100k steps")
    print(f"Weights: {best['weights']}")
    print(f"{'='*100}")

    print(f"\nDone! Results in {output_dir}/")
