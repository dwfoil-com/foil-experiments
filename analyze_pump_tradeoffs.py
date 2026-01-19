#!/usr/bin/env python3
"""Analyze pump frequency vs amplitude tradeoffs."""
import sys
sys.path.insert(0, "/Users/mattbook-air/claude-stuff/motion/foilpump")

import numpy as np
from foil_env.pump_foil_env_survival import PumpFoilEnvSurvival

FOIL_CONFIG = {
    'S': 0.08,  # 800 cm²
    'S_stab': 0.016,
    'stab_angle': -4.0,
    'AR': 8,
}

def test_frequency_amplitude(freq, amplitude, duration=10.0):
    """Run a test with fixed pump pattern and measure performance."""
    env = PumpFoilEnvSurvival(config=FOIL_CONFIG, reward_mode="distance")
    obs, _ = env.reset(seed=42)

    dt = env.dt
    steps = int(duration / dt)

    total_distance = 0.0
    total_energy = 0.0
    time = 0.0

    for _ in range(steps):
        # Sinusoidal leg pumping
        t = time
        target_leg = amplitude * np.sin(2 * np.pi * freq * t)
        # Velocity command to reach target
        leg_vel = 10.0 * (target_leg - (env.left_leg_pos + env.right_leg_pos) / 2)
        leg_action = leg_vel / env.MAX_LEG_VELOCITY

        # Fixed pitch, minimal arms/waist
        action = np.array([
            np.clip(leg_action, -1, 1),  # left leg
            np.clip(leg_action, -1, 1),  # right leg
            0.0, 0.0,  # arms
            0.4,  # pitch (fixed)
            0.0,  # waist
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        total_distance += env.state.vx * dt
        total_energy += info.get('power', 0) * dt
        time += dt

        if terminated:
            break

    final_vx = env.state.vx
    return {
        'freq': freq,
        'amplitude': amplitude,
        'duration': time,
        'distance': total_distance,
        'energy': total_energy,
        'final_vx': final_vx,
        'efficiency': total_distance / max(total_energy, 1),
        'reason': info.get('termination_reason', 'max_steps'),
    }

# Grid search over frequency and amplitude
print("=" * 70)
print("PUMP FREQUENCY vs AMPLITUDE ANALYSIS (800 cm² foil)")
print("=" * 70)
print()

freqs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
amps = [0.3, 0.5, 0.7, 0.9]  # fraction of max extension

results = []
for freq in freqs:
    for amp in amps:
        r = test_frequency_amplitude(freq, amp * 0.15)  # MAX_LEG_EXTENSION is 0.15
        results.append(r)
        print(f"Freq={freq:.1f}Hz Amp={amp:.0%}: {r['duration']:.1f}s, dist={r['distance']:.1f}m, "
              f"energy={r['energy']:.0f}J, vx_end={r['final_vx']:.2f}m/s, eff={r['efficiency']:.4f}, {r['reason']}")

print()
print("=" * 70)
print("BEST BY METRIC:")
print("=" * 70)

best_dist = max(results, key=lambda r: r['distance'])
best_eff = max(results, key=lambda r: r['efficiency'])
best_dur = max(results, key=lambda r: r['duration'])

print(f"Best distance:   {best_dist['freq']:.1f}Hz @ {best_dist['amplitude']/0.15:.0%} amp = {best_dist['distance']:.1f}m")
print(f"Best efficiency: {best_eff['freq']:.1f}Hz @ {best_eff['amplitude']/0.15:.0%} amp = {best_eff['efficiency']:.4f} m/J")
print(f"Best duration:   {best_dur['freq']:.1f}Hz @ {best_dur['amplitude']/0.15:.0%} amp = {best_dur['duration']:.1f}s")

# Energy analysis
print()
print("=" * 70)
print("ENERGY SCALING (theoretical vs measured):")
print("=" * 70)
base = results[0]  # 1.0 Hz, 0.3 amp
for r in results:
    if r['amplitude'] == base['amplitude']:
        freq_ratio = r['freq'] / base['freq']
        expected = freq_ratio ** 3  # A² × ω³, amplitude same
        actual = r['energy'] / max(base['energy'], 1)
        print(f"{r['freq']:.1f}Hz: expected {expected:.1f}x energy, actual {actual:.1f}x")
