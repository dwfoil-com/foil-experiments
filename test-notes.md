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
