#!/usr/bin/env python3
"""
Test full-amplitude same-phase arm swings and create comparison video.

Real pump foilers swing arms in big arcs crossing over the body.
Let's force this on the trained model and see what happens.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO
from foil_env.pump_foil_env_survival import PumpFoilEnvSurvival, analyze_pumping
from foil_env.foil_visualizer import draw_foil_and_rider, draw_water

FOIL_CONFIG = {
    'S': 0.08,  # 800 cm²
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}


def run_episode_with_frames(model, env, arm_mode="baseline", arm_amplitude=1.0, max_steps=600):
    """
    Run episode and collect frames for video.

    arm_mode:
        - "baseline": Use model's arm actions
        - "full_same": Full-amplitude same-phase arm swing
    arm_amplitude: 0-1 fraction of MAX_ARM_SWING to use
    """
    obs, _ = env.reset(seed=42)

    frames = []
    prev_leg_vel = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float32)

        # Get leg velocity and position to determine arm direction
        leg_vel = (env.left_leg_vel + env.right_leg_vel) / 2
        leg_pos = (env.left_leg_pos + env.right_leg_pos) / 2

        if arm_mode == "full_same":
            # Match model's strategy: MAX velocity in same direction as legs
            # No threshold - always command max, just switch direction
            arm_action = np.sign(leg_vel) * arm_amplitude if abs(leg_vel) > 0.01 else 0
            action[2] = arm_action
            action[3] = arm_action

        elif arm_mode == "full_opposite":
            # MAX velocity in opposite direction
            arm_action = -np.sign(leg_vel) * arm_amplitude if abs(leg_vel) > 0.01 else 0
            action[2] = arm_action
            action[3] = arm_action

        elif arm_mode == "bang_same":
            # Pure bang-bang: always ±1 based on leg velocity sign
            arm_action = np.sign(leg_vel) * arm_amplitude
            action[2] = arm_action
            action[3] = arm_action

        elif arm_mode == "bang_opposite":
            # Pure bang-bang opposite
            arm_action = -np.sign(leg_vel) * arm_amplitude
            action[2] = arm_action
            action[3] = arm_action

        elif arm_mode == "bang_lead":
            # Bang-bang same phase but using leg ACCELERATION to predict direction change
            # Arm leads leg by ~30ms - when leg is about to change direction, arm changes first
            leg_accel = (leg_vel - prev_leg_vel) / env.dt if step > 0 else 0

            # If leg velocity and acceleration have opposite signs, leg is about to reverse
            # Switch arm direction early based on where leg is GOING, not where it IS
            if abs(leg_vel) < 0.5 and np.sign(leg_accel) != 0:
                # Leg is slowing down - predict next direction from acceleration
                predicted_direction = np.sign(leg_accel)
            else:
                predicted_direction = np.sign(leg_vel) if abs(leg_vel) > 0.01 else 0

            arm_action = predicted_direction * arm_amplitude
            action[2] = arm_action
            action[3] = arm_action
            prev_leg_vel = leg_vel

        # Collect frame data before step
        frames.append({
            't': step * env.dt,
            'z': env.state.z,
            'vx': env.state.vx,
            'theta': env.state.theta,
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
            'waist': env.waist_pos,
        })

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # One more frame showing final state
            frames.append({
                't': (step + 1) * env.dt,
                'z': env.state.z,
                'vx': env.state.vx,
                'theta': env.state.theta,
                'left_leg': env.left_leg_pos,
                'right_leg': env.right_leg_pos,
                'left_arm': env.left_arm_pos,
                'right_arm': env.right_arm_pos,
                'waist': env.waist_pos,
            })
            break

    duration = len(frames) * env.dt
    reason = info.get('termination_reason', 'max_steps')

    # Analyze pumping
    leg_positions = [(f['left_leg'] + f['right_leg']) / 2 for f in frames]
    arm_positions = [(f['left_arm'] + f['right_arm']) / 2 for f in frames]
    pump_stats = analyze_pumping(leg_positions, env.dt)

    arm_range = max(arm_positions) - min(arm_positions) if arm_positions else 0

    return {
        'frames': frames,
        'duration': duration,
        'reason': reason,
        'pump_freq': pump_stats['freq'],
        'arm_range': arm_range,
        'final_vx': frames[-1]['vx'] if frames else 0,
    }


def create_comparison_video(baseline_data, fullswing_data, output_path, foil_config):
    """Create side-by-side comparison video."""

    frames_b = baseline_data['frames']
    frames_f = fullswing_data['frames']

    # Pad shorter one
    max_len = max(len(frames_b), len(frames_f))
    while len(frames_b) < max_len:
        frames_b.append(frames_b[-1].copy())
        frames_b[-1]['crashed'] = True
    while len(frames_f) < max_len:
        frames_f.append(frames_f[-1].copy())
        frames_f[-1]['crashed'] = True

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    frame_skip = 2
    render_indices = list(range(0, max_len, frame_skip))

    def animate(frame_num):
        i = render_indices[frame_num]

        for ax, frames, title, data in [
            (axes[0], frames_b, f"Baseline (model's arms)\n{baseline_data['duration']:.1f}s | {baseline_data['pump_freq']:.1f}Hz", baseline_data),
            (axes[1], frames_f, f"Full Swing (same-phase)\n{fullswing_data['duration']:.1f}s | {fullswing_data['pump_freq']:.1f}Hz", fullswing_data),
        ]:
            ax.clear()
            frame = frames[i]
            is_crashed = frame.get('crashed', False) or i >= len(data['frames']) - 1

            draw_water(ax, t=frame['t'], is_crashed=is_crashed, vx=frame['vx'])
            draw_foil_and_rider(
                ax, z=frame['z'], theta=frame['theta'],
                left_leg=frame['left_leg'], right_leg=frame['right_leg'],
                left_arm=frame['left_arm'], right_arm=frame['right_arm'],
                waist=frame['waist'], t=frame['t'],
                foil_config=foil_config, is_crashed=is_crashed,
                vx=frame['vx'], draw_water_bg=False,
            )
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.2, 2.0)
            ax.set_aspect('equal')
            ax.set_facecolor('lightcyan')
            ax.set_title(title, fontsize=11, fontweight='bold')

            # Show arm range
            arm_pos = (frame['left_arm'] + frame['right_arm']) / 2
            ax.text(-1.4, 1.8, f"t={frame['t']:.1f}s\narm={arm_pos:.2f}", fontsize=9)

        return []

    print(f"Creating video with {len(render_indices)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(render_indices), interval=33, blit=False)
    anim.save(output_path, writer='ffmpeg', fps=30)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading model...")
    model = PPO.load("checkpoints/combined1/model_100pct.zip")
    env = PumpFoilEnvSurvival(config=FOIL_CONFIG, reward_mode="distance")

    print("\n" + "="*70)
    print("FULL ARM SWING TEST")
    print("="*70)

    # Increase arm velocity for realistic swing
    original_arm_vel = env.MAX_ARM_VELOCITY
    env.MAX_ARM_VELOCITY = 12.0  # 12 rad/s = 687 deg/s (realistic for pumping)
    print(f"\nIncreased MAX_ARM_VELOCITY: {original_arm_vel} -> {env.MAX_ARM_VELOCITY} rad/s")

    # Test baseline
    print("\nRunning baseline (model's natural arms)...")
    baseline = run_episode_with_frames(model, env, arm_mode="baseline")
    print(f"  Duration: {baseline['duration']:.2f}s, Arm range: {np.degrees(baseline['arm_range']):.0f} deg")

    # Test bang-bang same (matching model's strategy)
    print("\nRunning BANG-BANG same-phase (±1 matching leg direction)...")
    bang_same = run_episode_with_frames(model, env, arm_mode="bang_same", arm_amplitude=1.0)
    print(f"  Duration: {bang_same['duration']:.2f}s, Arm range: {np.degrees(bang_same['arm_range']):.0f} deg")

    # Test bang-bang opposite
    print("\nRunning BANG-BANG opposite-phase...")
    bang_opp = run_episode_with_frames(model, env, arm_mode="bang_opposite", arm_amplitude=1.0)
    print(f"  Duration: {bang_opp['duration']:.2f}s, Arm range: {np.degrees(bang_opp['arm_range']):.0f} deg")

    # Test bang-bang with lead (using acceleration to predict)
    print("\nRunning BANG-BANG with 30ms LEAD (using acceleration)...")
    bang_lead = run_episode_with_frames(model, env, arm_mode="bang_lead", arm_amplitude=1.0)
    print(f"  Duration: {bang_lead['duration']:.2f}s, Arm range: {np.degrees(bang_lead['arm_range']):.0f} deg")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ARM SWING COMPARISON")
    print("="*70)
    print(f"{'Mode':<35} {'Duration':>10} {'Arm Range':>12} {'End':<15}")
    print("-"*70)
    print(f"{'Baseline (model learned)':<35} {baseline['duration']:>10.2f}s {np.degrees(baseline['arm_range']):>10.0f} deg {baseline['reason']:<15}")
    print(f"{'Bang-bang SAME phase':<35} {bang_same['duration']:>10.2f}s {np.degrees(bang_same['arm_range']):>10.0f} deg {bang_same['reason']:<15}")
    print(f"{'Bang-bang OPPOSITE phase':<35} {bang_opp['duration']:>10.2f}s {np.degrees(bang_opp['arm_range']):>10.0f} deg {bang_opp['reason']:<15}")
    print(f"{'Bang-bang LEAD (30ms early)':<35} {bang_lead['duration']:>10.2f}s {np.degrees(bang_lead['arm_range']):>10.0f} deg {bang_lead['reason']:<15}")

    print("\n" + "="*70)
    print("INSIGHT: Model arms LEAD legs by ~30ms")
    print("         Arms change direction BEFORE legs, optimizing force timing.")
    print("="*70)

    # Create comparison video
    print("\nCreating comparison video (baseline vs bang-bang same)...")
    create_comparison_video(baseline, bang_same, "arm_swing_comparison.mp4", FOIL_CONFIG)


if __name__ == "__main__":
    main()
