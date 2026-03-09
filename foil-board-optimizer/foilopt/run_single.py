"""
Run a single foil board topology optimization.

Usage:
    python -m foilopt.run_single [--config configs/default.yaml]
"""

import argparse
import os
import sys
import yaml
import numpy as np
import time

from .geometry.board import FoilBoard, create_default_load_cases
from .geometry.mesh import generate_hex_mesh
from .topology.simp import SIMPOptimizer, SIMPConfig
from .utils.export import export_density_to_stl


def main():
    parser = argparse.ArgumentParser(description="Foil board topology optimization")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--output", default="results/single_run", help="Output directory")
    parser.add_argument("--no-stl", action="store_true", help="Skip STL export")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config not found: {args.config}, using defaults")
        cfg = {}

    board_cfg = cfg.get("board", {})
    mesh_cfg = cfg.get("mesh", {})
    opt_cfg = cfg.get("optimization", {})

    # Setup board
    board = FoilBoard(
        length=board_cfg.get("length", 1.4),
        width=board_cfg.get("width", 0.5),
        thickness=board_cfg.get("thickness", 0.1),
        mast_mount_x=board_cfg.get("mast_mount_x", 0.45),
        mast_mount_y=board_cfg.get("mast_mount_y", 0.5),
    )

    # Generate mesh
    nelx = mesh_cfg.get("nelx", 28)
    nely = mesh_cfg.get("nely", 10)
    nelz = mesh_cfg.get("nelz", 4)

    print(f"Board: {board.length}m x {board.width}m x {board.thickness}m")
    print(f"Mesh: {nelx} x {nely} x {nelz} = {nelx*nely*nelz} elements")

    mesh = generate_hex_mesh(*board.get_domain_shape(), nelx, nely, nelz)
    print(f"Nodes: {mesh.n_nodes}, DOFs: {3 * mesh.n_nodes}")

    # Load cases
    load_cases = create_default_load_cases()
    print(f"Load cases: {[lc.name for lc in load_cases]}")

    # SIMP config
    simp_config = SIMPConfig(
        volfrac=opt_cfg.get("volfrac", 0.35),
        penal=opt_cfg.get("penal", 3.0),
        rmin=opt_cfg.get("rmin", 1.5),
        max_iter=opt_cfg.get("max_iter", 100),
        tol=opt_cfg.get("tol", 0.01),
        use_heaviside=opt_cfg.get("use_heaviside", False),
        move_limit=opt_cfg.get("move_limit", 0.2),
    )

    print(f"\nOptimization: volfrac={simp_config.volfrac}, "
          f"penal={simp_config.penal}, rmin={simp_config.rmin}")

    # Callback for progress
    def progress_callback(iteration, compliance, volume, change, density):
        if iteration % 10 == 0 or iteration < 5:
            print(f"  Iter {iteration:3d}: compliance={compliance:.4f}, "
                  f"vol={volume:.3f}, change={change:.4f}")

    # Run optimization
    print("\nStarting SIMP optimization...")
    t0 = time.time()

    optimizer = SIMPOptimizer(mesh, board, simp_config, callback=progress_callback)
    result = optimizer.optimize(load_cases)

    print(f"\nOptimization complete in {result.total_time:.1f}s")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Final compliance: {result.final_compliance:.4f}")
    print(f"  Final volume: {result.final_volume:.3f}")
    print(f"  Stiffness score: {result.stiffness_metrics.get('aggregate', {}).get('stiffness_score', 'N/A')}")

    # Save results
    os.makedirs(args.output, exist_ok=True)

    np.save(os.path.join(args.output, "density.npy"), result.density)
    print(f"\nDensity saved to: {args.output}/density.npy")

    if not args.no_stl:
        stl_path = os.path.join(args.output, "board_structure.stl")
        export_density_to_stl(result.density, mesh, threshold=0.5, output_path=stl_path)

    if not args.no_plot:
        try:
            from .utils.visualization import plot_convergence, plot_density_slices
            plot_convergence(result, save_path=os.path.join(args.output, "convergence.png"))
            plot_density_slices(result.density, mesh, save_path=os.path.join(args.output, "slices.png"))
            print(f"Plots saved to: {args.output}/")
        except ImportError:
            print("Matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
