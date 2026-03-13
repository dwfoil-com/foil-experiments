#!/usr/bin/env python3
"""Baseline simulation: standard EPS foam + carbon/epoxy foil board.

Computes weight and compliance for a conventional construction board
using the same FEA mesh and load cases as the topology optimizer,
to establish the target our optimized design needs to beat.

Material properties:
  - EPS foam core: E=5 MPa, rho=30 kg/m³
  - Carbon/epoxy shell (1.5mm): E=55 GPa, rho=1550 kg/m³
"""

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from foilopt.geometry.board import FoilBoard, LoadCase, load_board_shape
from foilopt.geometry.mesh import generate_hex_mesh
from foilopt.fea.solver import FEASolver3D


def compute_shell_mask(mesh, board, board_shape, shell_thickness_frac=0.12):
    """Identify elements that are part of the outer shell (vectorized)."""
    centers = mesh.element_centers()  # (N, 3)
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

    if board_shape:
        hw = board_shape.half_width_at(x)
        bot_z = board_shape.bot_z_at(x)
        deck_z = board_shape.deck_z_at(x)

        z_s3d = board_shape.z_min + (z / board.thickness) * (board_shape.z_max - board_shape.z_min)

        center_y = board.width / 2.0
        y_norm = np.abs(y - center_y) / np.maximum(hw, 1e-6)
        z_mid = (bot_z + deck_z) / 2.0
        z_half = (deck_z - bot_z) / 2.0
        z_norm = np.abs(z_s3d - z_mid) / np.maximum(z_half, 1e-6)

        r = np.power(y_norm, 2.5) + np.power(z_norm, 2.5)
        shell_mask = r > (1.0 - shell_thickness_frac)
    else:
        # Fallback: outermost layer
        nelx, nely, nelz = mesh.nelx, mesh.nely, mesh.nelz
        shell_mask = np.zeros(nelx * nely * nelz, dtype=bool)
        for k in range(nelz):
            for j in range(nely):
                for i in range(nelx):
                    if j == 0 or j == nely-1 or k == 0 or k == nelz-1:
                        shell_mask[k * nely * nelx + j * nelx + i] = True

    return shell_mask


def run_baseline():
    """Run forward FEA with EPS+carbon material properties."""
    meta_path = "results/modal_bulkhead5/meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    nelx, nely, nelz = meta["nelx"], meta["nely"], meta["nelz"]
    board = FoilBoard()
    board_shape = load_board_shape()

    print(f"Board: {board.length:.3f} x {board.width:.3f} x {board.thickness:.3f} m")
    print(f"Mesh: {nelx}×{nely}×{nelz} = {nelx*nely*nelz:,} elements")

    # Material properties
    EPS_E = 5.0e6       # 5 MPa — EPS foam
    EPS_rho = 30.0      # kg/m³
    CARBON_E = 55.0e9   # 55 GPa — carbon/epoxy quasi-isotropic
    CARBON_rho = 1550.0  # kg/m³

    # Create mesh and solver
    mesh = generate_hex_mesh(board.length, board.width, board.thickness, nelx, nely, nelz)
    solver = FEASolver3D(
        mesh, board, E0=CARBON_E, Emin=EPS_E, nu=0.35, penal=1.0,
        board_shape=board_shape
    )

    # Identify shell elements
    print("\nComputing shell mask...")
    shell_mask = compute_shell_mask(mesh, board, board_shape, shell_thickness_frac=0.12)
    n_shell = shell_mask.sum()
    n_core = (~shell_mask).sum()
    print(f"  Shell elements: {n_shell} ({n_shell/len(shell_mask)*100:.1f}%)")
    print(f"  Core elements:  {n_core} ({n_core/len(shell_mask)*100:.1f}%)")

    # Per-element material arrays
    n_elem = nelx * nely * nelz
    E0_arr = np.full(n_elem, EPS_E)
    E0_arr[shell_mask] = CARBON_E
    Emin_arr = np.full(n_elem, EPS_E * 0.01)
    Emin_arr[shell_mask] = EPS_E

    solver.set_material_arrays(E0_arr, Emin_arr)

    # Density = 1.0 everywhere — conventional solid board
    density = np.ones(n_elem, dtype=np.float64)

    # Real-world weight estimate
    planform_area = board.length * board.width * 0.65
    top_bottom = 2 * planform_area
    rail_perim = 2 * board.length * 0.85
    rails = rail_perim * board.thickness
    SA = top_bottom + rails

    real_shell_mass = SA * 0.0015 * CARBON_rho  # 1.5mm carbon
    board_volume = board.length * board.width * board.thickness * 0.65
    real_core_mass = board_volume * EPS_rho

    print(f"\n{'='*50}")
    print(f"  BASELINE: EPS + Carbon/Epoxy Foil Board")
    print(f"{'='*50}")
    print(f"  EPS foam core:      {real_core_mass:.2f} kg  ({EPS_rho} kg/m³, E={EPS_E/1e6:.0f} MPa)")
    print(f"  Carbon/epoxy shell: {real_shell_mass:.2f} kg  (1.5mm, {CARBON_rho} kg/m³, E={CARBON_E/1e9:.0f} GPa)")
    print(f"  Hardware/extras:    ~0.30 kg")
    print(f"  ─────────────────────────────")
    print(f"  Total:              {real_core_mass + real_shell_mass + 0.3:.2f} kg")
    print(f"{'='*50}")

    # Run FEA for each load case
    load_cases_meta = meta.get("load_cases", [])
    total_compliance = 0.0
    results = {}

    print(f"\nRunning {len(load_cases_meta)} load cases...")
    for lc in load_cases_meta:
        name = lc["name"]
        load = LoadCase(
            name=name,
            mast_force=tuple(lc["mast_force"]),
            mast_torque=tuple(lc["mast_torque"]),
        )

        t0 = time.time()
        u, info = solver.solve(density, load)
        elapsed = time.time() - t0

        compliance = info.get("compliance", float(solver.f @ u) if hasattr(solver, 'f') else 0)
        max_disp = float(np.max(np.abs(u)))

        print(f"  {name:20s}  C={compliance:.6f} J  max_disp={max_disp*1000:.3f} mm  ({elapsed:.1f}s)")

        results[name] = {
            "compliance": compliance,
            "max_displacement_mm": round(max_disp * 1000, 4),
        }
        total_compliance += compliance

    total_weight = real_core_mass + real_shell_mass + 0.3

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Total compliance: {total_compliance:.4f} J")
    print(f"  Board weight:     {total_weight:.2f} kg")
    print(f"")
    print(f"  Phase 1 optimized (for comparison):")
    print(f"    Compliance:     {meta.get('compliance', '?')} J")
    print(f"    Target mass:    {meta.get('target_mass_kg', '?')} kg")
    print(f"    Material:       Fiberglass E=20 GPa (single material)")
    print(f"{'='*50}")

    # Save
    output = {
        "type": "baseline_eps_carbon",
        "materials": {
            "core": {"name": "EPS foam", "E_Pa": EPS_E, "rho_kgm3": EPS_rho},
            "shell": {"name": "Carbon/epoxy", "E_Pa": CARBON_E, "rho_kgm3": CARBON_rho,
                       "thickness_mm": 1.5},
        },
        "weight": {
            "core_kg": round(real_core_mass, 3),
            "shell_kg": round(real_shell_mass, 3),
            "hardware_kg": 0.3,
            "total_kg": round(total_weight, 3),
        },
        "mesh": {"nelx": nelx, "nely": nely, "nelz": nelz},
        "load_cases": results,
        "total_compliance": round(total_compliance, 6),
        "comparison": {
            "phase1_compliance": meta.get("compliance"),
            "phase1_target_mass_kg": meta.get("target_mass_kg"),
            "phase1_material": "Fiberglass E=20 GPa (single material)",
        },
    }

    os.makedirs("results/baseline_eps", exist_ok=True)
    with open("results/baseline_eps/baseline_summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/baseline_eps/baseline_summary.json")


if __name__ == "__main__":
    run_baseline()
