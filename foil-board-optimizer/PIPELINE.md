# Foil Board Optimizer — Pipeline Reference

## Quick Glossary

| Term | What it means |
|------|--------------|
| **SIMP** | Solid Isotropic Material with Penalization — the optimization algorithm. Each element gets a density 0→1; penalty pushes toward binary (solid/void) |
| **Density field** | Array of per-element material fractions. 0.001 = void, 1.0 = solid. Shape depends on mesh resolution |
| **Compliance** | Total strain energy (Joules). Lower = stiffer board. The optimizer minimizes this |
| **Strain energy** | Per-element energy stored under load. High SE = that element is working hard |
| **Volfrac** | Volume fraction — what % of elements are solid. Controls weight |
| **Marching cubes** | Algorithm that converts a 3D density volume into a triangle mesh (STL) at a threshold |
| **Bulkhead** | Full-height transverse plate at a specific X position |
| **Shell** | Outer skin of the board (forced solid in optimizer, carbon fiber in reality) |

## File Formats

| Extension | What it is | How to view |
|-----------|-----------|-------------|
| `.npy` | NumPy array — density field or 3D volume | `python -c "import numpy as np; print(np.load('file.npy').shape)"` |
| `.bin` | Raw float32 binary — density or strain energy | Loaded by viewers, not human-readable |
| `.stl` | 3D triangle mesh | **Preview.app** (drag to rotate), or any 3D viewer |
| `.html` | Interactive Three.js viewer | Open in browser |
| `.png` | Static matplotlib visualization | Preview.app |
| `meta.json` | Optimization metadata (mesh size, compliance, load cases) | Any text editor |
| `summary.json` | Phase 2 bulkhead list with X positions | Any text editor |

## The Pipeline

```
Phase 1          Phase 2              Phase 3            Phase 4           Phase 5
3D Coarse    →   2D Cross-Sections →  3D Assembly     →  Shell + Internal → FEA Validation
modal_run.py     run_cross_sections   build_3d_structure  build_complete     validate_3d
                                                          _board             _structure
     ↓                ↓                    ↓                   ↓                 ↓
density.bin      density_x*.npy       board_continuous    complete_board     validation
meta.json        section_x*.png       _volume.npy         _fullboard.stl    _summary.json
strain_energy/   summary.json         board_continuous    board_volume.npy   validation
                                      .stl                                   _overview.png
                                      structure_overview
                                      .png
```

### Phase 1: 3D Topology Optimization (`modal_run.py`)

**What:** Runs coarse SIMP optimization on Modal GPU. Finds where material should go.

**Command:**
```bash
python modal_run.py --nelx 56 --nely 20 --nelz 12 --max-iter 100 \
  --target-mass 8.0 --output results/modal_bulkhead5
```

**Key inputs:** Board geometry (.s3dx), rider weight, 4-6 load cases
**Key outputs:**
- `density.bin` / `density.npy` — 56×20×12 = 13,440 element densities
- `meta.json` — mesh dims, compliance, load case forces
- `strain_energy/*.bin` — per-load-case energy maps (used by Phase 2)
- `board.stl` — coarse 3D mesh

**Time:** ~90 min on Modal, ~5400s local

---

### Phase 2: 2D Cross-Section Refinement (`run_cross_sections.py`)

**What:** Takes Phase 1's coarse "where" and runs high-res 2D optimization at each X-station to get fine "how" — the detailed rib/web pattern.

**Command:**
```bash
# Fullboard: all 56 slices at 100×30
python run_cross_sections.py --phase1 results/modal_bulkhead5 \
  --nely 100 --nelz 30 --volfrac 0.15 --output results/cross_sections_fullboard \
  --xmin 0.01 --xmax 1.63

# Hires: fewer slices at 200×60 (under-foot zone only)
python run_cross_sections.py --phase1 results/modal_bulkhead5 \
  --nely 200 --nelz 60 --volfrac 0.15 --output results/cross_sections_hires
```

**Key inputs:** Phase 1 `meta.json` + `strain_energy/` (for local load scaling)
**Key outputs:**
- `density_x{pos}.npy` — 2D density per slice (e.g. 100×30 or 200×60)
- `section_x{pos}.png` — visualization with ribs visible
- `summary.json` — list of bulkhead X positions

**Time:** ~2-5 min per slice, ~30 min for 56 slices

---

### Phase 3: 3D Structure Assembly (`build_3d_structure.py`)

**What:** Interpolates between 2D cross-sections to create a continuous 3D volume. Runs marching cubes to produce a printable STL.

**Command:**
```bash
python build_3d_structure.py --cross-sections results/cross_sections_fullboard \
  --output results/structure_fullboard.stl
```

**Key inputs:** Phase 2 `summary.json` + `density_x*.npy`
**Key outputs:**
- `*_volume.npy` — 3D voxel volume (e.g. 323×30×100). Axes: (X, Z, Y)
- `*.stl` — marching-cubes mesh. **Open in Preview.app for draggable 3D**
- `structure_overview.png` — 6-panel matplotlib (top/side/cross-sections/material%)

**Time:** ~30 seconds

---

### Phase 4: Complete Board Assembly (`build_complete_board.py`)

**What:** Merges outer shell (from .s3dx board shape) with internal structure from Phase 3.

**Command:**
```bash
# Continuous internal structure + shell
python build_complete_board.py --continuous \
  --cross-sections results/cross_sections_fullboard \
  --output results/complete_board_fullboard.stl

# Discrete bulkhead plates + shell
python build_complete_board.py --n-bulkheads 8 \
  --cross-sections results/cross_sections_fullboard \
  --output results/complete_board.stl
```

**Key inputs:** Phase 2 cross-sections + board shape (.s3dx)
**Key outputs:**
- `*.stl` — complete board: shell + internals
- `*_volume.npy` — full voxel volume
- `complete_board_overview.png` — top/side with bulkhead lines marked

**Time:** ~2 min

---

### Phase 5: Structural Validation (`validate_3d_structure.py`)

**What:** Closes the loop. Resamples the assembled board back onto the Phase 1 FEA mesh and runs forward solves to check compliance matches expectations.

**Command:**
```bash
python validate_3d_structure.py --phase1 results/modal_bulkhead5 \
  --board-volume results/structure_fullboard.npy \
  --output results/validation_fullboard
```

**Key inputs:** Phase 1 `meta.json` (mesh + load cases) + Phase 3/4 volume
**Key outputs:**
- `validation_summary.json` — per-load-case compliance + total
- `validation_overview.png` — strain energy comparison vs Phase 1
- `se_*.bin` — strain energy per load case (6 files)

**Time:** ~2 min

---

### Viewers

**Single result viewer:**
```bash
python build_viewer.py results/modal_bulkhead5 -o viewer.html
open viewer.html
```

**Comparison viewer (multiple datasets):**
```bash
python build_comparison_viewer.py -o comparison.html
open comparison.html
```

**Internal-only STL** (shell stripped, for Preview.app):
```bash
# Generated by stripping shell from board_continuous_volume.npy
# See board_internal_only.stl in results/
open -a Preview results/board_internal_only.stl
```

## Current Results Summary

| Result | Resolution | Solid% | Compliance | What |
|--------|-----------|--------|------------|------|
| `modal_bulkhead5/` | 56×20×12 | 22.5% | 0.053 | Phase 1 coarse SIMP |
| `cross_sections_fullboard/` | 56 × 100×30 | ~15% | varies | Phase 2 full board |
| `cross_sections_hires/` | 17 × 200×60 | ~15% | varies | Phase 2 under-foot (high detail) |
| `structure_fullboard.npy` | 323×30×100 | 18.1% | — | Phase 3 assembled volume |
| `board_continuous_volume.npy` | 329×30×100 | 16.6% | — | Phase 3 continuous (original) |
| `complete_board_fullboard.stl` | — | — | — | Phase 4 shell + continuous |
| `board_internal_only.stl` | — | 6.2% | — | Internal ribs only (no shell) |
| `validation_fullboard/` | resampled | 17.1% | 1.386J | Phase 5 verified |
