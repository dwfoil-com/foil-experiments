#!/usr/bin/env python3
"""
Analyze what forces the arms are actually generating.
"""

import numpy as np
from stable_baselines3 import PPO
from foil_env.pump_foil_env_survival import PumpFoilEnvSurvival

FOIL_CONFIG = {
    'S': 0.08,
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

# Constants from the env
ARM_MASS = 4.0  # kg per arm
ARM_MOMENT = 0.3  # m

def analyze_arm_contribution():
    print("Loading model...")
    model = PPO.load("checkpoints/combined1/model_100pct.zip")
    env = PumpFoilEnvSurvival(config=FOIL_CONFIG, reward_mode="distance")

    obs, _ = env.reset(seed=42)

    data = {
        't': [], 'z': [], 'theta': [],
        'leg_pos': [], 'leg_vel': [], 'leg_accel': [],
        'arm_pos': [], 'arm_vel': [], 'arm_accel': [],
        'leg_force': [], 'arm_vertical_force': [], 'arm_pitch_torque': [],
        'total_vertical': [], 'total_pitch': [],
    }

    prev_leg_vel = 0
    prev_arm_vel = 0

    for step in range(600):  # 6 seconds
        action, _ = model.predict(obs, deterministic=True)

        # Record before step
        leg_pos = (env.left_leg_pos + env.right_leg_pos) / 2
        arm_pos = (env.left_arm_pos + env.right_arm_pos) / 2
        leg_vel = (env.left_leg_vel + env.right_leg_vel) / 2
        arm_vel = (env.left_arm_vel + env.right_arm_vel) / 2

        obs, reward, terminated, truncated, info = env.step(action)

        # Compute accelerations
        new_leg_vel = (env.left_leg_vel + env.right_leg_vel) / 2
        new_arm_vel = (env.left_arm_vel + env.right_arm_vel) / 2
        leg_accel = (new_leg_vel - leg_vel) / env.dt
        arm_accel = (new_arm_vel - arm_vel) / env.dt

        # Compute forces (from env physics)
        # Leg force = -LEG_MASS * leg_accel (reaction force)
        leg_force = -env.LEG_MASS * 2 * leg_accel  # 2 legs

        # Arm vertical force (deweighting component)
        # arm_vertical_force = -ARM_MASS * arm_accel * cos(arm_pos)
        arm_vertical = -ARM_MASS * 2 * arm_accel * np.cos(arm_pos)

        # Arm pitch torque
        # arm_pitch_torque = -ARM_MASS * arm_accel * ARM_MOMENT
        arm_pitch = -ARM_MASS * 2 * arm_accel * ARM_MOMENT

        data['t'].append(step * env.dt)
        data['z'].append(env.state.z * 100)
        data['theta'].append(np.degrees(env.state.theta))
        data['leg_pos'].append(leg_pos * 100)
        data['leg_vel'].append(leg_vel)
        data['leg_accel'].append(leg_accel)
        data['arm_pos'].append(arm_pos * 100)
        data['arm_vel'].append(arm_vel)
        data['arm_accel'].append(arm_accel)
        data['leg_force'].append(leg_force)
        data['arm_vertical_force'].append(arm_vertical)
        data['arm_pitch_torque'].append(arm_pitch)
        data['total_vertical'].append(info.get('total_vertical_force', 0))
        data['total_pitch'].append(info.get('pitch_torque', 0))

        if terminated or truncated:
            break

    # Analysis
    print("\n" + "="*70)
    print("ARM FORCE ANALYSIS")
    print("="*70)

    leg_force = np.array(data['leg_force'])
    arm_vert = np.array(data['arm_vertical_force'])
    arm_pitch = np.array(data['arm_pitch_torque'])

    print(f"\nForce Statistics (first 5s):")
    print(f"  Leg vertical force:   mean={np.mean(leg_force):>8.1f}N,  std={np.std(leg_force):>8.1f}N,  range=[{np.min(leg_force):.0f}, {np.max(leg_force):.0f}]")
    print(f"  Arm vertical force:   mean={np.mean(arm_vert):>8.1f}N,  std={np.std(arm_vert):>8.1f}N,  range=[{np.min(arm_vert):.0f}, {np.max(arm_vert):.0f}]")
    print(f"  Arm pitch torque:     mean={np.mean(arm_pitch):>8.1f}Nm, std={np.std(arm_pitch):>8.1f}Nm, range=[{np.min(arm_pitch):.0f}, {np.max(arm_pitch):.0f}]")

    # Correlation between leg and arm
    leg_v = np.array(data['leg_vel'])
    arm_v = np.array(data['arm_vel'])
    corr = np.corrcoef(leg_v, arm_v)[0, 1]
    print(f"\n  Leg-Arm velocity correlation: {corr:.2f} ({'same phase' if corr > 0 else 'opposite phase'})")

    # What's the arm contribution to total force?
    print(f"\n  Arm vertical / Leg vertical ratio: {np.std(arm_vert) / np.std(leg_force):.1%}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    t = data['t']

    # Positions
    axes[0].plot(t, data['leg_pos'], 'b-', lw=2, label='Leg pos (cm)')
    axes[0].plot(t, data['arm_pos'], 'r-', lw=2, label='Arm pos (cm)')
    axes[0].set_ylabel('Position (cm)')
    axes[0].legend()
    axes[0].set_title('Model Arm Behavior Analysis')

    # Forces
    axes[1].plot(t, data['leg_force'], 'b-', lw=2, label='Leg force')
    axes[1].plot(t, data['arm_vertical_force'], 'r-', lw=2, label='Arm vertical')
    axes[1].axhline(y=0, color='k', ls=':')
    axes[1].set_ylabel('Vertical Force (N)')
    axes[1].legend()

    # Pitch torque
    axes[2].plot(t, data['arm_pitch_torque'], 'r-', lw=2, label='Arm pitch torque')
    axes[2].axhline(y=0, color='k', ls=':')
    axes[2].set_ylabel('Pitch Torque (Nm)')
    axes[2].legend()

    # State
    axes[3].plot(t, data['z'], 'b-', lw=2, label='Altitude (cm)')
    axes[3].plot(t, data['theta'], 'g-', lw=2, label='Pitch (deg)')
    axes[3].axhline(y=-50, color='r', ls=':', alpha=0.5)
    axes[3].set_ylabel('State')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('arm_forces.png', dpi=150)
    print(f"\nSaved: arm_forces.png")


if __name__ == "__main__":
    analyze_arm_contribution()
