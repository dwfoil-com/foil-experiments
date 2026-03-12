# Bulkhead Optimizer — Three-Phase Plan

## Context

The 3D SIMP optimizer (`results/modal_rib/`) at 29mm element size can't produce thin plates —
it blobs. The mast plate zone is a solid box. We need to force a manufacturable hollow-board
topology: transverse bulkheads + optional stringers.

Three phases, each building on the previous. All changes are additive to existing code.

---

## Completed State (before this plan)

- `foilopt/topology/simp.py`: SIMPOptimizer with weight constraint, MaxSolid, sensitivity averaging
- `foilopt/fea/solver.py`: FEASolver3D with board_shape-aware BCs
- `modal_run.py`: Modal runner, cpu=8, timeout=7200s, max_iter=100
- `results/modal_rib/`: 100-iter run, 8kg target, 56×20×12 mesh, 4 load cases
  - `density.bin`, `meta.json`, `strain_energy/{name}.bin`, `safety_factor.bin`
- `viewer.html`: 3D viewer with load-transfer visualization

---

## Mesh Element Ordering (critical for all phases)

From `generate_hex_mesh` loop order: `k` (nelz) slowest, `j` (nely) middle, `i` (nelx) fastest:

```
flat index e = k*(nely*nelx) + j*nelx + i
→ ix = e % nelx           (X-slice: 0..nelx-1)
→ iy = (e // nelx) % nely  (Y-position: 0..nely-1)
→ iz = e // (nelx*nely)    (Z-layer: 0..nelz-1)
```

`mesh.density_to_3d(density)` reshapes to `(nelx, nely, nelz)` correctly.

**For transverse bulkheads** (Y-Z plates at fixed X): group by `ix = e % nelx`.
Each X-slice has `nely*nelz = 240` elements at 56×20×12.

---

## Phase 1 — X-Column Design Variables (transverse bulkheads)

**Goal**: Force material to form full-height, full-width plates at discrete X positions.
The optimizer picks which X-slices get material (=where bulkheads go).
Each bulkhead spans full Y and Z within the board outline.

### What changes

Add `bulkhead_mode: bool = False` to `SIMPConfig`.

In `SIMPOptimizer.__init__` (when bulkhead_mode=True):
```python
self._ix_map = np.arange(mesh.n_elements) % mesh.nelx  # (n_elem,) → ix for each elem
self._n_free_per_slice = np.bincount(  # free elements per X-slice
    self._ix_map[~solid_mask & ~void_mask],
    minlength=mesh.nelx,
).astype(float)
```

New private methods:
```python
def _col_to_elem(self, x_col, solid_mask, void_mask):
    """Expand X-slice design vars to per-element density."""
    x = x_col[self._ix_map]
    x[solid_mask] = 1.0
    x[void_mask] = 0.001
    return x

def _elem_to_col_dc(self, dc_elem, solid_mask, void_mask):
    """Aggregate element sensitivities to X-slice design vars."""
    dc_free = dc_elem.copy()
    dc_free[solid_mask] = 0.0
    dc_free[void_mask] = 0.0
    return np.bincount(self._ix_map, weights=dc_free, minlength=self.mesh.nelx).astype(float)
```

In `optimize()` loop when bulkhead_mode=True:
- Design variable: `x_col` shape (nelx,), initialized at effective_volfrac
- Expand: `x_elem = _col_to_elem(x_col, solid_mask, void_mask)`
- FEA: `xPhys = density_filter(x_elem, H)` + Heaviside + re-enforce boundaries (unchanged)
- Compute `dc_elem` as usual (unchanged)
- Aggregate: `dc_col = _elem_to_col_dc(dc_elem, solid_mask, void_mask)`
- 1D filter in X: `dc_col = np.convolve(dc_col, np.ones(3)/3, mode='same')` (optional smoothing)
- OC on `x_col` with volume target: `sum(x_col * n_free_per_slice) / sum(n_free_per_slice)`

OC update mod — volume check uses weighted sum instead of mean:
```python
target_total = effective_volfrac * np.sum(n_free_per_slice)
if np.dot(xnew_col, n_free_per_slice) > target_total:
    l1 = lmid
else:
    l2 = lmid
```

### Running Phase 1

Add `--bulkhead-mode` flag to `modal_run.py`, output to `results/modal_bulkhead`.

Expected output: 10–20 filled X-slices (bulkhead positions) with ~8kg total mass.
Compliance will be higher than full 3D (fewer design degrees of freedom) but topology is
manufacturable: CNC-cut Y-Z cross-section profiles.

### What to look for in viewer

Slices with density > 0.5 = bulkhead positions. Count them, note clustering at:
- Mast mount region (highest strain energy)
- Under foot zones
- Near tail (torsion load path)

---

## Phase 2 — 2.5D Cross-Section Optimizer

**Goal**: For each bulkhead position found in Phase 1, find the optimal internal topology
of that bulkhead (where to cut holes, where to keep material) using a fine 2D mesh.

### New file: `foilopt/topology/cross_section.py`

```python
class CrossSectionOptimizer:
    """2D SIMP on Y-Z plane for a single board cross-section."""

    def __init__(self, x_pos, board_shape, mesh_2d, load_2d, config):
        # x_pos: position along board (meters)
        # board_shape: for Y-Z outline at that X
        # mesh_2d: fine 2D mesh (e.g. 80×30 elements for Y and Z)
        # load_2d: (n_elem_2d,) load weight array derived from 3D strain energy

    def optimize(self) -> np.ndarray:
        """Return 2D density field for this cross-section."""
        ...
```

### Load weighting from 3D results

```python
import numpy as np

# Load 3D strain energy for each load case
se_dir = "results/modal_rib/strain_energy"
load_cases = ["riding_normal", "pumping", "jump_landing", "carving"]

# For each X-slice ix, sum strain energy over all elements in that slice
# This tells us "how much work" each cross-section needs to carry
slice_load = np.zeros(nelx)  # (nelx,)
for name in load_cases:
    se = np.frombuffer(open(f"{se_dir}/{name}.bin", "rb").read(), dtype=np.float32)
    # sum se over elements in each X-slice
    for ix in range(nelx):
        slice_load[ix] += se[ix_map == ix].sum()

# Normalise per-slice load to drive 2D SIMP
```

### 2D SIMP details

- Domain: board Y-Z cross-section at each X, from `.s3dx` outline
- Mesh: 80×30 fine → 3.3mm×4mm elements → realistic 3-element ribs at 10mm
- Load: distributed compression (from deck) + shear (from adjacent bulkheads), weighted by `slice_load[ix]`
- Fixed BCs in 2D: bottom edge (hull skin attachment)
- Constraints: 2D volfrac matching target mass for that slice
- Output: 2D density → punch out holes in CAD → CNC bulkhead profile

### Runner

```bash
python run_cross_sections.py \
    --density results/modal_bulkhead/density.bin \
    --strain-energy results/modal_rib/strain_energy \
    --output results/cross_sections
```

Outputs per-slice PNG + assembled 3D STL of rib skeleton.

---

## Phase 3 — Longitudinal Stringers

**Goal**: Connect bulkheads with longitudinal stringers in the X direction.

### Option A: Rule-based (fast)

After Phase 2 bulkhead positions are known, add:
- 1 centerline stringer (Y=width/2, full length)
- 2 rail stringers (Y=width/4 and Y=3*width/4, full length)
- These intersect each bulkhead at known points → create cross joints

Implementation: force elements at those Y,Z positions to density=1.0 in Phase 1 (passive solid).

### Option B: X-direction SIMP pass (slower, optimal)

After Phase 2, run a separate 1D topology optimization along X for each Y,Z position:
- Design variables: which X-positions get material at this (Y,Z) cross-section
- Load: transverse loads between bulkheads (from Phase 2 local stress fields)
- Output: optimal stringer layout

### Mast box region

The mast box needs diagonal gussets (45° in X-Z plane) to resist foil torque:
- Detect high-torsion region (from carving/pumping strain energy)
- Add passive solid elements at 45° diagonal positions within ±3 X-slices of mast mount
- This is a manual override, not optimized

---

## Implementation Sequence

```
Phase 1 → Modal run → inspect viewer → Phase 2 → assemble → Phase 3 → final viewer
  ~2h             ~90min                ~1-2d               ~4h
```

Phase 1 is implemented first in this session. See `simp.py` for `bulkhead_mode` flag.
Phases 2 and 3 start from `plans/bulkhead-optimizer.md` in the next session.

---

## Key Files to Read at Session Start

1. `foilopt/geometry/mesh.py` — element ordering (ix = e % nelx)
2. `foilopt/topology/simp.py` — SIMPOptimizer, SIMPConfig
3. `results/modal_rib/meta.json` — mesh params, load case names
4. `results/modal_bulkhead/meta.json` — bulkhead run results (after Phase 1)
5. This file (`plans/bulkhead-optimizer.md`) — full spec

---

## Phase 1 Status

- [x] `SIMPConfig.bulkhead_mode` added
- [x] `_ix_map`, `_n_free_per_slice` computed in `__init__`
- [x] `_col_to_elem`, `_elem_to_col_dc`, `_oc_update_col` methods added
- [x] `optimize()` delegates to `_optimize_bulkhead()` when bulkhead_mode=True
- [x] OC update uses weighted volume check (`np.dot(xnew, n_free_per_slice) > target_total`)
- [x] `modal_run.py` `--bulkhead-mode` flag (auto-outputs to `results/modal_bulkhead`)
- [ ] Modal run launched → `results/modal_bulkhead`
