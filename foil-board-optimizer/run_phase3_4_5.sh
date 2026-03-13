#!/bin/bash
# Phase 3 (interpolate), Phase 4 (complete board), Phase 5 (3D validation) pipeline
# Runs on both cross_sections_hires and cross_sections_fullboard outputs

set -e
cd /Users/mattbook-air/foil-experiments/foil-board-optimizer

echo "=== PHASE 3: Building continuous 3D structures ==="
python build_3d_structure.py --cross-sections results/cross_sections_hires \
  --output results/structure_hires
echo "✓ Hires continuous structure complete"

python build_3d_structure.py --cross-sections results/cross_sections_fullboard \
  --output results/structure_fullboard
echo "✓ Fullboard continuous structure complete"

echo ""
echo "=== PHASE 4: Merging with outer shell ==="
python build_complete_board.py --continuous \
  --cross-sections results/cross_sections_hires \
  --output results/complete_board_hires.stl
echo "✓ Hires complete board (shell + internal structure)"

python build_complete_board.py --continuous \
  --cross-sections results/cross_sections_fullboard \
  --output results/complete_board_fullboard.stl
echo "✓ Fullboard complete board (shell + internal structure)"

echo ""
echo "=== PHASE 5: 3D structural validation ==="
python validate_3d_structure.py \
  --phase1 results/modal_bulkhead5 \
  --board-volume results/structure_hires/board_continuous_volume.npy \
  --output results/validation_hires
echo "✓ Hires validation complete"

python validate_3d_structure.py \
  --phase1 results/modal_bulkhead5 \
  --board-volume results/structure_fullboard/board_continuous_volume.npy \
  --output results/validation_fullboard
echo "✓ Fullboard validation complete"

echo ""
echo "=== COMPLETE ==="
echo "All phases complete. Outputs:"
echo "  - Continuous structures: results/structure_{hires,fullboard}/"
echo "  - Complete boards (shell+core): results/complete_board_{hires,fullboard}.stl"
echo "  - Validation summaries: results/validation_{hires,fullboard}/"
echo ""
echo "Next: Compare validation_hires vs validation_fullboard compliance/displacement"
