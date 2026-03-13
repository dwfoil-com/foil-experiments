#!/bin/bash
# Auto-run Phase 3/4/5 on hires when Phase 2 completes

cd /Users/mattbook-air/foil-experiments/foil-board-optimizer

echo "=== HIRES Phase 3: Building continuous 3D structure ==="
python build_3d_structure.py --cross-sections results/cross_sections_hires \
  --output results/structure_hires && echo "✓ Phase 3 complete"

echo ""
echo "=== HIRES Phase 4: Merging with outer shell ==="
python build_complete_board.py --continuous \
  --cross-sections results/cross_sections_hires \
  --output results/complete_board_hires.stl && echo "✓ Phase 4 complete"

echo ""
echo "=== HIRES Phase 5: 3D Structural Validation ==="
python validate_3d_structure.py \
  --phase1 results/modal_bulkhead5 \
  --board-volume results/structure_hires.npy \
  --output results/validation_hires && echo "✓ Phase 5 complete"

echo ""
echo "=== HIRES COMPLETE ==="
echo "Outputs:"
echo "  - results/complete_board_hires.stl (fine-res internal structure, under-foot zone)"
echo "  - results/validation_hires/validation_summary.json (compliance per load case)"
echo ""
echo "COMPARISON: fullboard vs hires"
echo "  Fullboard: 56 slices, 100×30 resolution, nose-to-tail coverage, 17.1% solid → 1.386J compliance"
echo "  Hires:     23 slices, 200×60 resolution, under-foot only, TBD% solid → TBD J compliance"
