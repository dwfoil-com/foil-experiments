# Test Notes

Notable observations from training runs.

## smoke_test2 (30k steps, 2026-01-19)
- **Observation**: Agents figure out to pump and stall
- **Checkpoints**: 1%, 25%, 50%, 75%, 100%
- **Video**: smoke_test2_evolution.mp4

## run1 (1M steps, 2026-01-19)
- **Config**: Modal A10G GPU, 8 parallel envs, curriculum_phase=2
- **Result**: 9.2s flight, 78% leg range, 1.7Hz pump frequency
- **Termination**: board_touchdown
- **Video**: checkpoints/run1/

## run2 (1M steps, resumed from run1, 2026-01-19)
- **Config**: Same as run1, resumed from run1/model_100pct.zip
- **Result**: 10.0s flight, 77% leg range, 1.7Hz pump frequency
- **Termination**: max_steps (no crash - reached video limit)
- **Video**: run2_evolution.mp4
- **Progress**: +0.8s flight time, model now survives full 10s evaluation

## ent005_2M - Best Pure RL (2026-01-19)
- **Config**: 2M steps, ent_coef=0.005, increased velocity limits (ARM 12 rad/s, LEG 2.0 m/s)
- **Result**: 5.0s flight, 2.3Hz pump, 152° arm range
- **Arm-leg correlation**: r=+0.44 (same-phase)
- **Termination**: energy_exhausted (good - means survived long)

---

## ACMPC / CPG Experiments (2026-01-20)

Explored structured approaches inspired by ACMPC (Actor-Critic MPC) research.

### hybrid_v2 - ACMPC-style planner
- **Approach**: MPC-like trajectory planner + RL learns pump parameters
- **Result**: 2.8s flight, r=0.00 correlation, 158° arm range, 2.2Hz
- **Problem**: Planner too rigid, couldn't adapt to altitude changes

### cpg_v1 - Anti-phase CPG
- **Approach**: Central Pattern Generator with arms opposite to legs
- **Result**: 1.1s flight, r=-0.37 (anti-phase), dV=+0.59 m/s
- **Problem**: Anti-phase doesn't generate lift, crashes immediately

### cpg_v2_samephase - Same-phase CPG
- **Approach**: Flipped to arms same-phase as legs (matching pure RL discovery)
- **Result**: 1.0s flight, r=+0.36, dV=+0.51 m/s
- **Problem**: Still crashes - phase wasn't the issue

### cpg_v3_freedom - CPG with altitude offset + stronger residuals
- **Approach**: Added altitude-responsive DC offset, 2x residual strength
- **Result**: 1.0s flight, r=+0.12, dV=+0.70 m/s
- **Problem**: Still crashes - sinusoidal structure fundamentally too rigid

### Key Finding
**CPG structure doesn't help this task.** All CPG variants crash in ~1s despite:
- Good pump frequency (2.2-2.4 Hz)
- Positive velocity change (accelerating!)
- Various phase relationships

Pure RL (5.0s) succeeds because it can make arbitrary corrections each timestep.
The sinusoidal CPG constraint prevents real-time altitude recovery.

| Approach | Flight | Arm-Leg r | Notes |
|----------|--------|-----------|-------|
| Pure RL (ent005_2M) | **5.0s** | +0.44 | Best result |
| Hybrid ACMPC | 2.8s | 0.00 | Too constrained |
| CPG anti-phase | 1.1s | -0.37 | Wrong phase |
| CPG same-phase | 1.0s | +0.36 | Phase not the issue |
| CPG + freedom | 1.0s | +0.12 | Structure too rigid |

**Conclusion**: For pump foil, pure RL's flexibility > structured approaches.

---

## Fixes Applied

### Motion Trails Restored (2026-01-19)
- Added `compute_body_positions()` function to foil_visualizer.py
- Added `draw_motion_trails()` function to foil_visualizer.py
- Updated create_video.py to pre-compute body positions and draw trails
- Trails show fading paths for: feet (blue/red), hands (blue/red), head (orange)
- Trail length: 15 frames, offset by velocity to stream behind

### Physics/Visualization Alignment (2026-01-19)
- Fixed board position formula: `board_y = (mast_length - riding_depth) + z`
- Operating range now equals mast length (70cm)
- At z=-50cm (touchdown): board correctly at water surface
- At z=+20cm (breach): foil at water surface
- Added limit lines to video plots (breach, touchdown, stall velocity)

---

## TODO: Settings to Validate

### Starting Position (FIXED)
- **Changed**: TARGET_ALTITUDE from 0.15 → 0.10
- **Now**: 10cm to breach, 60cm to touchdown (balanced margins)

### Other Settings to Check
- [ ] Rider mass (70kg) - realistic?
- [ ] Board mass (6kg) - realistic for foil board?
- [ ] Wing area (0.18 m²) - matches real foils?
- [ ] Aspect ratio (8) - typical for foil wings?
- [ ] Stall angle (13°) - correct for hydrofoils?
- [ ] Mast length (70cm) - standard size?
- [ ] Riding depth (20cm) - typical wing depth?
- [ ] Pump frequency target (2Hz) - matches real pumping?
- [ ] Max leg extension (15cm) - realistic human movement?
- [ ] Max arm swing (1.5 rad = 86°) - realistic?
- [ ] Energy budget (6000J) - how many pumps is this?

---

## Missing Capabilities

### Forward/Back Movement on Board
**Status**: NOT IMPLEMENTED

The current model does not support moving forward and back on the board. The rider's foot position is fixed at `STANCE_WIDTH = 0.30` with symmetric placement around the board center (line 256 in pump_foil_env_curriculum.py).

To add this dimension, would need:
1. New action dimension for fore/aft stance shift
2. New state variables for foot x-positions on board
3. Update body_model.py to accept variable foot positions
4. Update foil_visualizer.py to render shifted stance
