"""Phase 5: 3D structural validation of the assembled complete board.

Resamples the assembled board volume onto the Phase 1 FEA mesh and runs
a forward solve with each load case. Reports per-load-case compliance,
per-element strain energy, and flags regions where the interpolated
internal structure carries load poorly.

Interpolation between cross-sections is not the last structural word —
this script closes the loop.

Usage:
    python validate_3d_structure.py
    python validate_3d_structure.py --board-volume results/complete_board/board_continuous_volume.npy
    python validate_3d_structure.py --phase1 results/modal_bulkhead_v2 --board-volume results/complete_board/board_continuous_volume.npy
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def resample_volume_to_mesh(volume, vol_x_coords, vol_ly, vol_lz, mesh, threshold=0.5):
    """Resample assembled voxel volume onto Phase 1 FEA mesh element centres.

    Returns density array (n_elements,) in Phase 1 mesh ordering.
    """
    nx_v, nelz_v, nely_v = volume.shape
    x_min_v = float(vol_x_coords[0])
    x_max_v = float(vol_x_coords[-1])
    lx_v = x_max_v - x_min_v
    ly_v = float(vol_ly)
    lz_v = float(vol_lz)

    # Phase 1 mesh element centres (nelx × nely × nelz ordering)
    nelx, nely, nelz = mesh.nelx, mesh.nely, mesh.nelz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    # Flat index: e = k*(nelx*nely) + j*nelx + i  (i=X, j=Y, k=Z)
    e_i = np.arange(mesh.n_elements) % nelx
    e_j = (np.arange(mesh.n_elements) // nelx) % nely
    e_k = np.arange(mesh.n_elements) // (nelx * nely)

    xc = (e_i + 0.5) * dx
    yc = (e_j + 0.5) * dy
    zc = (e_k + 0.5) * dz

    # Map to voxel indices (nearest-neighbour)
    ix = np.clip(((xc - x_min_v) / lx_v * (nx_v - 1)).astype(int), 0, nx_v - 1)
    iy = np.clip((yc / ly_v * (nely_v - 1)).astype(int), 0, nely_v - 1)
    iz = np.clip((zc / lz_v * (nelz_v - 1)).astype(int), 0, nelz_v - 1)

    # volume axes: (X, Z, Y) — match build_complete_board convention
    density = volume[ix, iz, iy].astype(np.float32)
    return density


def load_volume(vol_path):
    """Load volume .npy and try to infer x_coords from sibling meta."""
    volume = np.load(vol_path)   # (nx, nelz, nely)
    nx, nelz, nely = volume.shape

    # Try to infer x_coords and physical dims from cross-sections summary
    cs_dir = os.path.dirname(vol_path)
    summary_path = os.path.join(cs_dir, "summary.json")

    from foilopt.geometry.board import FoilBoard
    board = FoilBoard()

    if os.path.exists(summary_path):
        import json as _json
        summary = _json.load(open(summary_path))
        bhs = summary.get("bulkheads", [])
        if bhs:
            x_min = bhs[0]["x_pos"]
            x_max = bhs[-1]["x_pos"]
        else:
            x_min, x_max = 0.0, board.length
    else:
        # Fall back: assume volume spans full board starting at X=0
        x_min, x_max = 0.0, board.length

    x_coords = np.linspace(x_min, x_max, nx)
    return volume, x_coords, board.width, board.thickness


def run_validation(phase1_dir, volume, vol_x_coords, vol_ly, vol_lz, output_dir):
    from foilopt.geometry.board import FoilBoard, create_default_load_cases, load_board_shape
    from foilopt.geometry.mesh import generate_hex_mesh
    from foilopt.fea.solver import FEASolver3D

    meta_path = os.path.join(phase1_dir, "meta.json")
    meta = json.load(open(meta_path))
    nelx, nely, nelz = meta["nelx"], meta["nely"], meta["nelz"]

    board = FoilBoard()
    board_shape = load_board_shape()
    mesh = generate_hex_mesh(*board.get_domain_shape(), nelx, nely, nelz)
    solver = FEASolver3D(mesh, board, board_shape=board_shape)

    load_cases = create_default_load_cases()

    print(f"Phase 1 mesh: {nelx}×{nely}×{nelz} = {mesh.n_elements} elements")
    print(f"Volume shape: {volume.shape}  X=[{vol_x_coords[0]:.3f},{vol_x_coords[-1]:.3f}]m")

    density = resample_volume_to_mesh(volume, vol_x_coords, vol_ly, vol_lz, mesh)
    density = np.array(density, dtype=np.float64)   # solver expects float64
    solid_pct = (density > 0.5).mean() * 100
    print(f"Resampled density: solid>{0.5}: {solid_pct:.1f}%  "
          f"mean={density.mean():.3f}  max={density.max():.3f}")

    # Save resampled density for comparison with Phase 1
    np.save(os.path.join(output_dir, "resampled_density.npy"), density)

    results = {}
    strain_energy_total = np.zeros(mesh.n_elements, dtype=np.float32)

    for lc in load_cases:
        print(f"\n  Load case: {lc.name}")
        u, info = solver.solve(density, lc)
        ce = solver.compute_element_compliance(density, u)
        E_elem = solver.Emin + density**solver.penal * (solver.E0 - solver.Emin)
        se = (E_elem * ce).astype(np.float32)
        strain_energy_total += se

        results[lc.name] = {
            "compliance": float(info["compliance"]),
            "max_displacement": float(info["max_displacement"]),
        }
        print(f"    compliance={info['compliance']:.4f}  "
              f"max_disp={info['max_displacement']*1000:.1f}mm")
        se.tofile(os.path.join(output_dir, f"se_{lc.name}.bin"))

    # Aggregate: compliance per X-slice (flag weak interpolated regions)
    e_i = np.arange(mesh.n_elements) % nelx
    slice_se = np.array([strain_energy_total[e_i == i].sum() for i in range(nelx)])
    slice_se_norm = slice_se / (slice_se.max() + 1e-12)

    # Compare to Phase 1 slice SE if available
    phase1_slice_se = None
    phase1_se_dir = os.path.join(phase1_dir, "strain_energy")
    if os.path.exists(phase1_se_dir):
        from run_cross_sections import compute_slice_loads
        phase1_slice_se = compute_slice_loads(phase1_se_dir, meta)
        phase1_slice_se /= (phase1_slice_se.max() + 1e-12)

    # Save summary
    summary = {
        "phase1_dir": phase1_dir,
        "load_cases": results,
        "total_compliance": sum(r["compliance"] for r in results.values()),
        "solid_fraction": float(solid_pct / 100),
    }
    with open(os.path.join(output_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal compliance: {summary['total_compliance']:.4f}")

    _save_validation_plot(
        slice_se_norm, phase1_slice_se, density, mesh,
        vol_x_coords, nelx, results, output_dir,
    )
    return summary


def _save_validation_plot(slice_se, phase1_se, density, mesh,
                          vol_x_coords, nelx, results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lx = mesh.nelx * mesh.dx
        x_slice = np.linspace(0, lx, nelx) * 1000

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Strain energy per slice: assembled vs Phase 1
        axes[0].plot(x_slice, slice_se, "b-", label="Assembled (this validation)", linewidth=1.5)
        if phase1_se is not None:
            axes[0].plot(x_slice, phase1_se, "r--", label="Phase 1 (original)", linewidth=1.5, alpha=0.7)
        axes[0].set_title("Normalised strain energy per X-slice")
        axes[0].set_xlabel("X (mm)"); axes[0].set_ylabel("Normalised SE")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Density distribution along X
        e_i = np.arange(mesh.n_elements) % nelx
        slice_density = np.array([density[e_i == i].mean() for i in range(nelx)])
        axes[1].plot(x_slice, slice_density * 100, "g-", linewidth=1.5)
        axes[1].set_title("Mean density per X-slice")
        axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Mean density (%)")
        axes[1].grid(True, alpha=0.3)

        # Compliance bar chart per load case
        lc_names = list(results.keys())
        compliances = [results[n]["compliance"] for n in lc_names]
        axes[2].bar(range(len(lc_names)), compliances, color="steelblue", alpha=0.8)
        axes[2].set_xticks(range(len(lc_names)))
        axes[2].set_xticklabels(lc_names, rotation=20, ha="right")
        axes[2].set_title("Compliance by load case")
        axes[2].set_ylabel("Compliance (J)"); axes[2].grid(True, alpha=0.3, axis="y")

        fig.suptitle("3D Structural Validation — Assembled Board", fontsize=13)
        fig.tight_layout()
        out = os.path.join(output_dir, "validation_overview.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Validation overview: {out}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="3D structural validation of assembled board")
    parser.add_argument("--phase1", default="results/modal_bulkhead5",
                        help="Phase 1 results directory (for mesh params and load cases)")
    parser.add_argument("--board-volume",
                        default="results/complete_board/board_continuous_volume.npy",
                        help="Assembled board volume .npy from build_complete_board.py")
    parser.add_argument("--output", default="results/validation",
                        help="Output directory for validation results")
    args = parser.parse_args()

    if not os.path.exists(args.board_volume):
        # Try sibling .npy files
        candidates = [
            "results/complete_board/board_continuous_volume.npy",
            "results/complete_board/board_volume.npy",
            "results/cross_sections_dense/structure_volume.npy",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.board_volume = c
                print(f"Using volume: {c}")
                break
        else:
            print(f"No board volume found. Run build_complete_board.py --continuous first.")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading volume: {args.board_volume}")
    volume, x_coords, ly, lz = load_volume(args.board_volume)
    print(f"Volume: {volume.shape}  solid>{0.5}: {(volume>0.5).mean()*100:.1f}%")

    run_validation(args.phase1, volume, x_coords, ly, lz, args.output)
    print(f"\nValidation complete. Results in {args.output}/")


if __name__ == "__main__":
    main()
