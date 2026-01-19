#!/usr/bin/env python3
"""
Test different arm patterns with the trained model.

Patterns:
1. Baseline (model's natural arm behavior)
2. Arms opposite to legs (deweighting theory)
3. Arms same phase as legs (anti-deweighting)
4. Arms fixed at neutral
"""

import numpy as np
from stable_baselines3 import PPO
from foil_env.pump_foil_env_survival import PumpFoilEnvSurvival, analyze_pumping

FOIL_CONFIG = {
    'S': 0.08,  # 800 cm²
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

def run_episode(model, env, arm_mode="baseline", max_steps=1000):
    """
    Run one episode with specified arm pattern.

    arm_mode:
        - "baseline": Use model's arm actions
        - "opposite": Arms swing opposite to legs
        - "same": Arms swing same phase as legs
        - "fixed": Arms stay at neutral
    """
    obs, _ = env.reset(seed=42)

    results = {
        'times': [], 'z': [], 'vx': [],
        'leg_pos': [], 'arm_pos': [],
        'leg_vel': [], 'arm_vel': [],
    }

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float32)

        # Override arm actions based on mode
        if arm_mode == "opposite":
            # Arms opposite to legs: when legs extend (positive vel), arms retract (negative vel)
            leg_vel_avg = (env.left_leg_vel + env.right_leg_vel) / 2
            arm_action = -leg_vel_avg / env.MAX_LEG_VELOCITY  # Opposite, scaled
            action[2] = np.clip(arm_action, -1, 1)
            action[3] = np.clip(arm_action, -1, 1)

        elif arm_mode == "same":
            # Arms same phase as legs
            leg_vel_avg = (env.left_leg_vel + env.right_leg_vel) / 2
            arm_action = leg_vel_avg / env.MAX_LEG_VELOCITY
            action[2] = np.clip(arm_action, -1, 1)
            action[3] = np.clip(arm_action, -1, 1)

        elif arm_mode == "fixed":
            # Arms stay neutral
            action[2] = 0.0
            action[3] = 0.0
        # else: baseline - use model's actions

        obs, reward, terminated, truncated, info = env.step(action)

        results['times'].append(step * env.dt)
        results['z'].append(env.state.z)
        results['vx'].append(env.state.vx)
        results['leg_pos'].append((env.left_leg_pos + env.right_leg_pos) / 2)
        results['arm_pos'].append((env.left_arm_pos + env.right_arm_pos) / 2)
        results['leg_vel'].append((env.left_leg_vel + env.right_leg_vel) / 2)
        results['arm_vel'].append((env.left_arm_vel + env.right_arm_vel) / 2)

        if terminated or truncated:
            break

    # Analysis
    duration = len(results['times']) * env.dt
    final_vx = results['vx'][-1] if results['vx'] else 0
    mean_vx = np.mean(results['vx']) if results['vx'] else 0

    pump_stats = analyze_pumping(results['leg_pos'], env.dt)

    # Arm-leg correlation
    if len(results['leg_vel']) > 20:
        leg_v = np.array(results['leg_vel'])
        arm_v = np.array(results['arm_vel'])
        if np.std(leg_v) > 0 and np.std(arm_v) > 0:
            correlation = np.corrcoef(leg_v, arm_v)[0, 1]
        else:
            correlation = 0
    else:
        correlation = 0

    arm_range = max(results['arm_pos']) - min(results['arm_pos']) if results['arm_pos'] else 0

    return {
        'duration': duration,
        'final_vx': final_vx,
        'mean_vx': mean_vx,
        'pump_freq': pump_stats['freq'],
        'leg_range': pump_stats['range_pct'],
        'arm_range_cm': arm_range * 100,
        'arm_leg_corr': correlation,  # Negative = opposite motion
        'reason': info.get('termination_reason', 'max_steps'),
        'raw': results,
    }


def main():
    print("Loading model...")
    model = PPO.load("checkpoints/combined1/model_100pct.zip")

    env = PumpFoilEnvSurvival(config=FOIL_CONFIG, reward_mode="distance")

    modes = ["baseline", "opposite", "same", "fixed"]

    print()
    print("=" * 80)
    print("ARM PATTERN COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Mode':<12} {'Duration':>10} {'Mean Vx':>10} {'Final Vx':>10} {'Pump Hz':>10} {'Arm Range':>12} {'Arm-Leg r':>12} {'End':<15}")
    print("-" * 80)

    all_results = {}
    for mode in modes:
        r = run_episode(model, env, arm_mode=mode, max_steps=1000)
        all_results[mode] = r
        print(f"{mode:<12} {r['duration']:>10.2f}s {r['mean_vx']:>10.2f} {r['final_vx']:>10.2f} {r['pump_freq']:>10.1f} {r['arm_range_cm']:>10.1f}cm {r['arm_leg_corr']:>12.2f} {r['reason']:<15}")

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print()
    print("Arm-Leg correlation:")
    print("  r < 0: Arms move OPPOSITE to legs (deweighting)")
    print("  r > 0: Arms move SAME as legs")
    print("  r ≈ 0: No coordination")
    print()

    # Compare
    baseline = all_results['baseline']
    opposite = all_results['opposite']

    if opposite['duration'] > baseline['duration']:
        print(f"✓ OPPOSITE arms improved duration: {baseline['duration']:.2f}s → {opposite['duration']:.2f}s (+{opposite['duration']-baseline['duration']:.2f}s)")
    else:
        print(f"✗ OPPOSITE arms hurt duration: {baseline['duration']:.2f}s → {opposite['duration']:.2f}s ({opposite['duration']-baseline['duration']:.2f}s)")

    if opposite['mean_vx'] > baseline['mean_vx']:
        print(f"✓ OPPOSITE arms improved velocity: {baseline['mean_vx']:.2f} → {opposite['mean_vx']:.2f} m/s")
    else:
        print(f"✗ OPPOSITE arms hurt velocity: {baseline['mean_vx']:.2f} → {opposite['mean_vx']:.2f} m/s")

    return all_results


def plot_comparison(all_results):
    """Plot time series comparison."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    modes = ["baseline", "opposite", "fixed"]
    colors = {"baseline": "blue", "opposite": "red", "same": "orange", "fixed": "gray"}

    for mode in modes:
        r = all_results[mode]
        t = r['raw']['times']
        axes[0].plot(t, np.array(r['raw']['z']) * 100, color=colors[mode], label=mode, lw=2)
        axes[1].plot(t, r['raw']['vx'], color=colors[mode], label=mode, lw=2)

        # Overlay leg and arm positions
        leg = np.array(r['raw']['leg_pos']) * 100
        arm = np.array(r['raw']['arm_pos']) * 100
        axes[2].plot(t, leg, color=colors[mode], lw=2, label=f"{mode} leg")
        axes[2].plot(t, arm, color=colors[mode], lw=1, ls='--', alpha=0.7, label=f"{mode} arm")

    axes[0].set_ylabel("Altitude (cm)")
    axes[0].legend()
    axes[0].axhline(y=-50, color='red', ls=':', alpha=0.5)
    axes[0].set_title("Arm Pattern Comparison")

    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].axhline(y=4.04, color='red', ls=':', alpha=0.5, label='stall')

    axes[2].set_ylabel("Leg/Arm position (cm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncol=3, fontsize=8)

    plt.tight_layout()
    plt.savefig("arm_comparison.png", dpi=150)
    print("\nSaved: arm_comparison.png")
    plt.close()


if __name__ == "__main__":
    all_results = main()
    if all_results:
        plot_comparison(all_results)
