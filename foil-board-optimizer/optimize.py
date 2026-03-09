#!/usr/bin/env python3
"""
Foil Board Structure Optimization — The Experiment File.

This is the file that the autonomous agent modifies each experiment.
It runs one complete topology optimization and logs the result.

Modify the CONFIGURATION section below to try different parameters.
Run: python optimize.py
"""

import sys
import os
import time
import csv
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from foilopt.geometry.board import FoilBoard, create_default_load_cases, LoadCase
from foilopt.geometry.mesh import generate_hex_mesh
from foilopt.topology.simp import SIMPOptimizer, SIMPConfig
from foilopt.utils.export import export_density_to_stl

# ============================================================================
# CONFIGURATION — The agent modifies this section each experiment
# ============================================================================

# Mesh resolution (higher = more accurate but slower)
NELX = 28       # elements along board length
NELY = 10       # elements along board width
NELZ = 4        # elements through board thickness

# Board geometry (meters)
BOARD_LENGTH = 1.40
BOARD_WIDTH = 0.50
BOARD_THICKNESS = 0.10

# Material
E0 = 2.0e9      # Young's modulus (Pa) — PETG
NU = 0.35        # Poisson's ratio

# SIMP optimization parameters
VOLFRAC = 0.35   # target volume fraction (35% material, 65% void)
PENAL = 3.0      # penalization power
RMIN = 1.5       # filter radius (x max element size)
MAX_ITER = 100   # max optimization iterations
TOL = 0.01       # convergence tolerance
USE_HEAVISIDE = False  # Heaviside projection for crisp boundaries
MOVE_LIMIT = 0.2      # max density change per iteration

# Load cases to optimize against (choose from the list below)
# Available: "riding_normal", "pumping", "jump_landing", "carving"
LOAD_CASE_NAMES = ["riding_normal", "pumping", "jump_landing", "carving"]

# Rider weight (kg)
RIDER_WEIGHT = 80.0

# Experiment note — describe what you're trying
NOTE = "baseline: default parameters, all four load cases"

# ============================================================================
# END CONFIGURATION — Don't modify below unless you know what you're doing
# ============================================================================

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")


def run_experiment():
    """Run one topology optimization experiment and log results."""
    t_start = time.time()

    # Setup board
    board = FoilBoard(
        length=BOARD_LENGTH,
        width=BOARD_WIDTH,
        thickness=BOARD_THICKNESS,
    )

    # Generate mesh
    mesh = generate_hex_mesh(
        *board.get_domain_shape(),
        NELX, NELY, NELZ,
    )

    n_elements = mesh.n_elements
    n_dofs = 3 * mesh.n_nodes
    print(f"Mesh: {NELX}x{NELY}x{NELZ} = {n_elements} elements, {n_dofs} DOFs")

    # Setup load cases
    all_cases = {lc.name: lc for lc in create_default_load_cases()}
    load_cases = []
    for name in LOAD_CASE_NAMES:
        if name in all_cases:
            lc = all_cases[name]
            lc.weight_rider_kg = RIDER_WEIGHT
            load_cases.append(lc)

    if not load_cases:
        load_cases = create_default_load_cases()

    print(f"Load cases: {[lc.name for lc in load_cases]}")

    # Configure SIMP
    simp_config = SIMPConfig(
        volfrac=VOLFRAC,
        penal=PENAL,
        rmin=RMIN,
        max_iter=MAX_ITER,
        tol=TOL,
        use_heaviside=USE_HEAVISIDE,
        move_limit=MOVE_LIMIT,
    )

    print(f"Config: vf={VOLFRAC}, p={PENAL}, rmin={RMIN}, "
          f"heaviside={USE_HEAVISIDE}, move={MOVE_LIMIT}")

    # Progress callback
    def progress(iteration, compliance, volume, change, density):
        if iteration % 10 == 0:
            print(f"  iter {iteration:3d}: C={compliance:.6f} V={volume:.4f} "
                  f"change={change:.5f}")

    # Run optimization
    print("\nOptimizing...")
    optimizer = SIMPOptimizer(
        mesh, board, simp_config, callback=progress
    )
    result = optimizer.optimize(load_cases)

    t_total = time.time() - t_start

    # Extract results
    compliance = result.final_compliance
    volume = result.final_volume
    n_iters = result.n_iterations
    converged = result.converged
    max_disp = result.stiffness_metrics.get("aggregate", {}).get(
        "max_displacement", float("inf")
    )
    stiffness_score = result.stiffness_metrics.get("aggregate", {}).get(
        "stiffness_score", 0.0
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULT: compliance={compliance:.6f}  volume={volume:.4f}  "
          f"max_disp={max_disp:.8f}")
    print(f"        stiffness_score={stiffness_score:.6f}  "
          f"iters={n_iters}  converged={converged}  time={t_total:.1f}s")
    print(f"{'='*60}")

    # Export STL
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stl_path = os.path.join(results_dir, f"board_{timestamp}.stl")

    export_density_to_stl(result.density, mesh, threshold=0.5, output_path=stl_path)

    # Save density
    npy_path = os.path.join(results_dir, f"density_{timestamp}.npy")
    np.save(npy_path, result.density)

    # Log to results.tsv
    header = [
        "timestamp", "compliance", "volume", "stiffness_score", "max_disp",
        "nelx", "nely", "nelz", "volfrac", "penal", "rmin",
        "heaviside", "move_limit", "max_iter", "iters", "converged",
        "E0", "rider_kg", "load_cases", "time_s", "note",
    ]

    row = [
        datetime.now().isoformat(),
        f"{compliance:.8f}",
        f"{volume:.4f}",
        f"{stiffness_score:.6f}",
        f"{max_disp:.10f}",
        NELX, NELY, NELZ,
        VOLFRAC, PENAL, RMIN,
        USE_HEAVISIDE, MOVE_LIMIT, MAX_ITER,
        n_iters, converged,
        E0, RIDER_WEIGHT,
        "+".join(LOAD_CASE_NAMES),
        f"{t_total:.1f}",
        NOTE,
    ]

    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"\nResults appended to: {RESULTS_FILE}")
    print(f"STL exported to: {stl_path}")
    print(f"Density saved to: {npy_path}")

    return compliance


if __name__ == "__main__":
    compliance = run_experiment()
