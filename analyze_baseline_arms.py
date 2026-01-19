#!/usr/bin/env python3
"""Detailed analysis of baseline arm behavior."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from foil_env.pump_foil_env_survival import PumpFoilEnvSurvival

FOIL_CONFIG = {
    'S': 0.08,
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

def main():
    print("Loading model...")
    model = PPO.load("checkpoints/combined1/model_100pct.zip")
    env = PumpFoilEnvSurvival(config=FOIL_CONFIG, reward_mode="distance")

    obs, _ = env.reset(seed=42)

    data = {
        't': [], 'z': [],
        'leg_pos': [], 'leg_vel': [],
        'arm_pos': [], 'arm_vel': [],
        'arm_action': [],  # What the model actually commands
    }

    for step in range(600):
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action)

        data['t'].append(step * env.dt)
        data['z'].append(env.state.z * 100)
        data['leg_pos'].append((env.left_leg_pos + env.right_leg_pos) / 2 * 100)
        data['leg_vel'].append((env.left_leg_vel + env.right_leg_vel) / 2)
        data['arm_pos'].append((env.left_arm_pos + env.right_arm_pos) / 2)
        data['arm_vel'].append((env.left_arm_vel + env.right_arm_vel) / 2)
        data['arm_action'].append((action[2] + action[3]) / 2)  # Model's arm action

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Correlation analysis
    leg_vel = np.array(data['leg_vel'])
    arm_vel = np.array(data['arm_vel'])
    arm_action = np.array(data['arm_action'])

    corr_vel = np.corrcoef(leg_vel, arm_vel)[0, 1]
    corr_action = np.corrcoef(leg_vel, arm_action)[0, 1]

    print(f"\nLeg-Arm velocity correlation: {corr_vel:.2f}")
    print(f"Leg vel - Arm action correlation: {corr_action:.2f}")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    t = data['t']

    # Altitude
    axes[0].plot(t, data['z'], 'b-', lw=2)
    axes[0].axhline(y=-50, color='r', ls=':', alpha=0.5)
    axes[0].set_ylabel('Altitude (cm)')
    axes[0].set_title(f"Baseline Model Arm Analysis (r={corr_vel:.2f})")

    # Leg and arm positions overlaid
    axes[1].plot(t, data['leg_pos'], 'b-', lw=2, label='Leg pos (cm)')
    ax1b = axes[1].twinx()
    ax1b.plot(t, np.array(data['arm_pos']) * 57.3, 'r-', lw=2, label='Arm pos (deg)')  # Convert to degrees
    axes[1].set_ylabel('Leg (cm)', color='blue')
    ax1b.set_ylabel('Arm (deg)', color='red')
    axes[1].legend(loc='upper left')
    ax1b.legend(loc='upper right')

    # Velocities overlaid
    axes[2].plot(t, data['leg_vel'], 'b-', lw=2, label='Leg vel')
    axes[2].plot(t, data['arm_vel'], 'r-', lw=2, label='Arm vel')
    axes[2].axhline(y=0, color='k', ls=':')
    axes[2].set_ylabel('Velocity')
    axes[2].legend()

    # Arm action commanded
    axes[3].plot(t, data['arm_action'], 'g-', lw=2, label='Arm action (model)')
    axes[3].axhline(y=0, color='k', ls=':')
    axes[3].set_ylabel('Arm Action')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('baseline_arm_analysis.png', dpi=150)
    print(f"\nSaved: baseline_arm_analysis.png")

    # Additional insight: when does the model command high arm velocity?
    high_arm_action = np.abs(arm_action) > 0.5
    high_leg_vel = np.abs(leg_vel) > 0.3

    print(f"\nModel commands high arm action: {100*np.mean(high_arm_action):.1f}% of time")
    print(f"High leg velocity: {100*np.mean(high_leg_vel):.1f}% of time")

    # When arm action is high, what's the leg doing?
    if np.any(high_arm_action):
        avg_leg_vel_when_arm_high = np.mean(leg_vel[high_arm_action])
        print(f"Avg leg vel when arm action high: {avg_leg_vel_when_arm_high:.2f}")

    # Cross-correlation to find timing offset (using numpy)
    corr = np.correlate(arm_action - np.mean(arm_action), leg_vel - np.mean(leg_vel), mode='same')
    lags = np.arange(-len(leg_vel)//2, len(leg_vel)//2)
    best_lag = lags[np.argmax(corr)]
    print(f"\nBest lag (arm leads leg): {best_lag} samples ({best_lag * 0.01:.3f}s)")

    # Phase relationship
    # Find zero-crossings of leg_vel
    leg_crossings = np.where(np.diff(np.signbit(leg_vel)))[0]
    arm_crossings = np.where(np.diff(np.signbit(arm_action)))[0]

    print(f"\nLeg vel zero crossings: {len(leg_crossings)}")
    print(f"Arm action zero crossings: {len(arm_crossings)}")

    if len(leg_crossings) > 2 and len(arm_crossings) > 2:
        # Compare crossing times
        leg_cross_times = np.array(leg_crossings) * 0.01
        arm_cross_times = np.array(arm_crossings) * 0.01

        # Find closest arm crossing to each leg crossing
        phase_diffs = []
        for lc in leg_cross_times[:10]:
            closest_arm = arm_cross_times[np.argmin(np.abs(arm_cross_times - lc))]
            phase_diffs.append(closest_arm - lc)

        avg_phase_diff = np.mean(phase_diffs)
        print(f"Avg phase difference (arm - leg): {avg_phase_diff*1000:.1f}ms")


if __name__ == "__main__":
    main()
