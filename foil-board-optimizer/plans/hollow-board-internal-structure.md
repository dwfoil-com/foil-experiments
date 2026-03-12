# Hollow Board Internal Structure — Design Plan

## Status

- [x] **Phase 1** — 3D SIMP in bulkhead mode: finds transverse rib positions (`modal_run.py --bulkhead-mode`)
- [x] **Phase 2** — 2D cross-section SIMP at each X-slice at 5mm×4mm resolution (`run_cross_sections.py`)
- [x] **Phase 3** — Interpolate slices → continuous 3D STL (`build_3d_structure.py`)
- [x] **Phase 4** — Outer shell + internal structure merged into complete board STL (`build_complete_board.py`)

---

## Next Steps

### 1. Full-resolution Phase 2 on Modal *(most immediate)*

The cross-sections ran at 100×30 (5mm×4mm elements) — coarse enough that you lose thin ribs and arch details. At 200×60 (2.5mm×2mm) you'd get genuine fine-grain topology: thin-wall ribs, true lightening holes, diagonal tension members. That's the resolution where it starts looking like a Divergent Technologies print.

Run Phase 2 on Modal the same way Phase 1 runs — parallelize all 23 slices simultaneously rather than sequentially.

- [ ] Add Modal function to `run_cross_sections.py` that fans out slices as parallel remote calls
- [ ] Run at `--nely 200 --nelz 60 --max-iter 150`
- [ ] Rebuild complete board with high-res cross-sections

### 2. ~~Merge internal structure with outer shell~~ ✓ Done

`build_complete_board.py` handles this — outer shell + internal structure in one STL, both discrete-plate and continuous modes.

### 3. Extend beyond the foot zone

Active zone is currently X=0.65–1.31m (under-foot). Nose and tail are hollow. At low volfrac (10–15%), running Phase 2 on the full board would add structural contribution — especially the tail block where the mast track is, and the nose which takes impact loads.

- [ ] Run `run_cross_sections.py` with `--xmin 0 --xmax 1.64` (full board)
- [ ] Use lower `--volfrac 0.12` outside the foot zone
- [ ] Rebuild complete board

### 4. Manufacturability constraints

- [ ] Minimum member thickness — so slicer doesn't thin features below nozzle diameter
- [ ] Maximum overhang angle — avoid supports inside the board cavity
- [ ] Infill connection points — ensure internal structure bonds to shell (print adhesion)

These can be implemented as post-processing filters on the density field before marching cubes, or as additional constraints in the SIMP loop.

### 5. Fiber-reinforced shell + printed core hybrid

The interesting product design: 3D-printed internal lattice in carbon-filled nylon or PETG, wrapped with prepreg carbon on the outside. The optimizer currently treats shell and core as the same material. Separating them (shell = high-stiffness carbon E=70GPa, core = moderate-stiffness print E=5GPa) would change the load path significantly — the core topology would likely open up more aggressively and route loads differently.

- [ ] Add dual-material mode to `CrossSectionConfig`: separate `E0_shell` and `E0_core`
- [ ] Shell elements use high E, free elements use print material E
- [ ] Re-run and compare topology against current single-material result

---

## Background: Why the 2.5D Approach Was Right

The original plan evaluated four options:

| Option | Verdict |
|--------|---------|
| A: Tune SIMP + MaxSolid | Implemented — MaxSolid disabled in bulkhead mode, sensitivity averaging added |
| B: Increase 3D mesh resolution | Impractical — 2.6M elements, 50+ hrs/run |
| C: 2.5D cross-section optimizer | **Chosen** — implemented as Phase 2, works well |
| D: Moving Morphable Components | Not needed — cross-section approach gives clean plate-like output |

The cross-section approach produces directly CNC-cuttable or 3D-printable bulkhead profiles at each X position. The continuous interpolation between slices (Phase 3) gives the smooth 3D structure needed for a printed core.

---

## Physical Reality: What Is a Hollow Foil Board?

```
Cross-section (rear foot zone):
┌─────────────────────────────┐  ← deck skin (0.5–1mm carbon)
│   rib    rib    rib    rib  │  ← transverse bulkheads (~4mm carbon)
│                             │
│         ←air→               │
└─────────────────────────────┘  ← hull skin (0.5–1mm carbon)
         ↑ stringer
```

Key structural members:
- **Deck/hull skins** — modelled as the forced-solid shell
- **Transverse bulkheads** — resist rider weight, local crushing
- **Longitudinal stringers** — resist nose-to-tail bending + torsion
- **Diagonal ribs near mast** — resist torsional loads from foil

---

## References

- Fernández et al. (2020): *Imposing min/max member size, cavity size, separation distance* — CMAME, arXiv:2003.00263
- Guest (2009): *Imposing maximum length scale* — Struct. Multidisc. Optim.
- Lazarov, Wang & Sigmund (2016): *Length scale and manufacturability in density-based TO* — Arch. Applied Mechanics
