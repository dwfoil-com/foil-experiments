#!/usr/bin/env python3
"""
Test body visualization to verify leg, arm, and waist movements are shown correctly.

Creates a side-by-side comparison showing:
- Left: Full crouch position
- Middle: Neutral position
- Right: Full extension position

Plus a video showing smooth transitions through the full range of motion.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from foil_env.foil_visualizer import draw_foil_and_rider, draw_water

FOIL_CONFIG = {
    'S': 0.08,
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

# Physical limits from curriculum env
MAX_LEG_EXT = 0.15  # meters
MAX_ARM_SWING = 1.5  # radians (~86 deg)
MAX_WAIST_ANGLE = 0.3  # radians (~17 deg)


def create_static_comparison():
    """Create static image comparing crouch, neutral, and extended positions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    positions = [
        ("Full Crouch", -1.0, -MAX_ARM_SWING, -MAX_WAIST_ANGLE),
        ("Neutral", 0.0, 0.0, 0.0),
        ("Full Extend", 1.0, MAX_ARM_SWING, MAX_WAIST_ANGLE),
    ]

    for ax, (title, leg_norm, arm_rad, waist_rad) in zip(axes, positions):
        ax.clear()

        # Convert normalized leg to meters
        leg_m = leg_norm * MAX_LEG_EXT

        draw_water(ax, t=0, is_crashed=False, vx=4.5)
        draw_foil_and_rider(
            ax, z=0.1, theta=0.05,
            left_leg=leg_m, right_leg=leg_m,
            left_arm=arm_rad, right_arm=arm_rad,
            waist=waist_rad, t=0,
            foil_config=FOIL_CONFIG, is_crashed=False,
            max_leg_ext=MAX_LEG_EXT, max_arm_swing=MAX_ARM_SWING,
            vx=4.5, draw_water_bg=False,
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 2.2)
        ax.set_aspect('equal')
        ax.set_facecolor('lightcyan')
        ax.set_title(f"{title}\nLeg: {leg_m*100:.0f}cm, Arm: {np.degrees(arm_rad):.0f}°, Waist: {np.degrees(waist_rad):.0f}°",
                     fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('body_visualization_test.png', dpi=150)
    print("Saved: body_visualization_test.png")
    plt.close()


def create_sweep_video():
    """Create video sweeping through full range of motion."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 5 seconds at 30fps = 150 frames
    # Sweep from -1 to +1 and back
    n_frames = 150

    def animate(frame):
        ax.clear()

        # Sinusoidal sweep for smooth motion
        t = frame / 30.0  # time in seconds

        # Legs at 2 Hz
        leg_phase = 2 * np.pi * 2.0 * t
        leg_norm = np.sin(leg_phase)
        leg_m = leg_norm * MAX_LEG_EXT

        # Arms at 2 Hz (same phase to test same-phase motion)
        arm_phase = leg_phase
        arm_rad = np.sin(arm_phase) * MAX_ARM_SWING

        # Waist at 1 Hz (slower rocking)
        waist_phase = 2 * np.pi * 1.0 * t
        waist_rad = np.sin(waist_phase) * MAX_WAIST_ANGLE

        draw_water(ax, t=t, is_crashed=False, vx=4.5)
        draw_foil_and_rider(
            ax, z=0.1, theta=0.05,
            left_leg=leg_m, right_leg=leg_m,
            left_arm=arm_rad, right_arm=arm_rad,
            waist=waist_rad, t=t,
            foil_config=FOIL_CONFIG, is_crashed=False,
            max_leg_ext=MAX_LEG_EXT, max_arm_swing=MAX_ARM_SWING,
            vx=4.5, draw_water_bg=False,
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 2.2)
        ax.set_aspect('equal')
        ax.set_facecolor('lightcyan')
        ax.set_title(f"Body Motion Test (t={t:.2f}s)\n"
                     f"Leg: {leg_m*100:+.1f}cm | Arm: {np.degrees(arm_rad):+.0f}° | Waist: {np.degrees(waist_rad):+.0f}°",
                     fontsize=11, fontweight='bold')

        return []

    print(f"Creating sweep video with {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=33, blit=False)
    anim.save('body_motion_test.mp4', writer='ffmpeg', fps=30)
    plt.close()
    print("Saved: body_motion_test.mp4")


def print_geometry_info():
    """Print expected vs actual visual ranges."""
    from foil_env.foil_visualizer import RIDER_GEOMETRY

    geom = RIDER_GEOMETRY
    shin = geom['shin_len']
    thigh = geom['thigh_len']

    # New formula
    base_stand = shin + thigh * 0.75
    crouch_height = base_stand - MAX_LEG_EXT
    extend_height = base_stand + MAX_LEG_EXT

    print("\n" + "="*60)
    print("BODY VISUALIZATION GEOMETRY")
    print("="*60)

    print(f"\nLEGS:")
    print(f"  Shin length: {shin*100:.0f}cm")
    print(f"  Thigh length: {thigh*100:.0f}cm")
    print(f"  MAX_LEG_EXTENSION: ±{MAX_LEG_EXT*100:.0f}cm")
    print(f"  Base standing height: {base_stand*100:.0f}cm")
    print(f"  Full crouch hip height: {crouch_height*100:.0f}cm")
    print(f"  Full extend hip height: {extend_height*100:.0f}cm")
    print(f"  Visual range: {(extend_height - crouch_height)*100:.0f}cm (should match {MAX_LEG_EXT*2*100:.0f}cm)")

    print(f"\nARMS:")
    print(f"  Upper arm: {geom['upper_arm_len']*100:.0f}cm")
    print(f"  Forearm: {geom['forearm_len']*100:.0f}cm")
    print(f"  MAX_ARM_SWING: ±{np.degrees(MAX_ARM_SWING):.0f}°")
    print(f"  Arm moves directly in radians (no scaling)")

    print(f"\nWAIST:")
    print(f"  Torso length: {geom['torso_len']*100:.0f}cm")
    print(f"  MAX_WAIST_ANGLE: ±{np.degrees(MAX_WAIST_ANGLE):.0f}°")
    print(f"  Waist uses raw radians (no scaling)")

    print("="*60)


def main():
    print_geometry_info()
    create_static_comparison()
    create_sweep_video()

    print("\nTest complete! Check:")
    print("  - body_visualization_test.png (static comparison)")
    print("  - body_motion_test.mp4 (smooth motion sweep)")


if __name__ == "__main__":
    main()
