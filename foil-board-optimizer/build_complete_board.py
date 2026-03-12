"""Phase 4: Assemble outer shell + N discrete internal bulkheads → complete board STL.

Combines:
  - Outer shell: full-length board skin from the .s3dx geometry
  - Internal bulkheads: N evenly-spaced optimized cross-sections as thin plates
    (representing carbon/glass bulkhead plates to be replaced with continuous
     3D-printed structure in a later step)

Usage:
    python build_complete_board.py
    python build_complete_board.py --cross-sections results/cross_sections_dense --n-bulkheads 10
    python build_complete_board.py --plate-thickness 4 --output results/complete_board.stl
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def load_cross_sections(cs_dir: str):
    """Load cross-sections from Phase 2 output directory."""
    summary_path = os.path.join(cs_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.json in {cs_dir}")
    summary = json.load(open(summary_path))
    sections = []
    for bh in summary["bulkheads"]:
        x_pos = bh["x_pos"]
        npy = os.path.join(cs_dir, f"density_x{x_pos:.3f}.npy")
        if os.path.exists(npy):
            d = np.load(npy)  # (nelz, nely)
            sections.append((x_pos, d))
    sections.sort(key=lambda s: s[0])
    print(f"Loaded {len(sections)} cross-sections from {cs_dir}")
    return sections, summary["config"]


def _interp_section(sections_x, sections_d, x):
    """Linearly interpolate between two adjacent cross-sections at position x."""
    if x <= sections_x[0]:
        return sections_d[0]
    if x >= sections_x[-1]:
        return sections_d[-1]
    i_hi = int(np.searchsorted(sections_x, x))
    i_lo = i_hi - 1
    t = (x - sections_x[i_lo]) / (sections_x[i_hi] - sections_x[i_lo])
    return (1.0 - t) * sections_d[i_lo] + t * sections_d[i_hi]


def build_volume(
    board,
    board_shape,
    sections,
    n_bulkheads: int = 10,
    plate_thickness_mm: float = 4.0,
    dx_mm: float = 5.0,
    shell_frac: float = 0.10,
    continuous: bool = False,
):
    """Build full-board voxel volume: shell everywhere + internal structure.

    Two modes:
      continuous=False  — N discrete plates at evenly-spaced X positions
      continuous=True   — linearly interpolated structure across the full active zone

    Returns:
        volume: (nx, nelz, nely) float32
        x_coords: X positions (m)
        ly, lz: board width / thickness (m)
        bh_x: bulkhead X positions (for overview plot)
    """
    lx = board.length
    ly = board.width
    lz = board.thickness

    dx = dx_mm / 1000.0
    x_coords = np.arange(0, lx + dx * 0.5, dx)
    nx = len(x_coords)

    nelz, nely = sections[0][1].shape
    dy = ly / nely
    dz = lz / nelz

    n_elem = nely * nelz
    e_j = np.arange(n_elem) % nely
    e_k = np.arange(n_elem) // nely
    y_c = (e_j + 0.5) * dy
    z_c = (e_k + 0.5) * dz

    # Active zone X bounds from sections
    x_min_active = sections[0][0]
    x_max_active = sections[-1][0]

    # Precompute interpolation arrays for continuous mode
    sec_x = np.array([s[0] for s in sections])
    sec_d = np.stack([s[1] for s in sections], axis=0)   # (n, nelz, nely)

    # For discrete mode: pick N evenly-spaced plates
    n = len(sections)
    idxs = np.round(np.linspace(0, n - 1, min(n_bulkheads, n))).astype(int)
    bulkheads = [sections[i] for i in idxs]
    bh_x = np.array([b[0] for b in bulkheads])
    half_plate = plate_thickness_mm / 1000.0 / 2.0

    mode_str = "continuous interpolation" if continuous else f"{len(bulkheads)} discrete plates ({plate_thickness_mm:.0f}mm thick)"
    print(f"\nBoard: {lx*1000:.0f}mm × {ly*1000:.0f}mm × {lz*1000:.0f}mm")
    print(f"Volume: {nx}×{nelz}×{nely}  dx={dx_mm:.0f}mm  dy={dy*1000:.1f}mm  dz={dz*1000:.1f}mm")
    print(f"Mode: {mode_str}")
    print(f"Active zone: X={x_min_active*1000:.0f}–{x_max_active*1000:.0f}mm")
    if not continuous:
        print(f"Plates at X (mm): {[f'{x*1000:.0f}' for x in bh_x]}")

    volume = np.zeros((nx, nelz, nely), dtype=np.float32)

    for ix, x in enumerate(x_coords):
        if ix % 50 == 0:
            print(f"  Voxelising X={x*1000:.0f}mm ({ix}/{nx})")

        x_arr = np.full(n_elem, x)
        inside = board_shape.is_inside(x_arr, y_c, z_c, lz)
        shell = board_shape.is_on_shell(x_arr, y_c, z_c, lz, thickness=shell_frac) & inside

        density = np.where(shell, 1.0, np.where(inside, 0.001, 0.0))

        if continuous and x_min_active <= x <= x_max_active:
            # Interpolate internal structure from cross-sections
            interp = _interp_section(sec_x, sec_d, x).ravel()
            density = np.maximum(density, interp)
            density[~inside] = 0.0
        elif not continuous:
            # Insert nearest plate if within half-thickness
            dists = np.abs(x - bh_x)
            nearest = int(np.argmin(dists))
            if dists[nearest] <= half_plate:
                bh_flat = bulkheads[nearest][1].ravel()
                density = np.maximum(density, bh_flat)
                density[~inside] = 0.0

        volume[ix] = density.reshape(nelz, nely)

    solid_pct = (volume > 0.5).mean() * 100
    print(f"\nSolid fraction (>0.5): {solid_pct:.1f}%")
    return volume, x_coords, ly, lz, bh_x
    return volume, x_coords, ly, lz


def save_overview(volume, x_coords, ly, lz, bh_x, output_dir):
    """Save top/side views with bulkhead positions marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nx, nelz, nely = volume.shape
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        # Top view (X-Y, max over Z)
        top = volume.max(axis=1)
        axes[0].imshow(top.T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1,
                       extent=[x_coords[0]*1000, x_coords[-1]*1000, 0, ly*1000])
        for bx in bh_x:
            axes[0].axvline(bx*1000, color="red", linewidth=0.8, alpha=0.7)
        axes[0].set_title("Top view (max over Z) — red = bulkheads")
        axes[0].set_xlabel("X (mm)"); axes[0].set_ylabel("Y (mm)")

        # Side view (X-Z, max over Y)
        side = volume.max(axis=2)
        axes[1].imshow(side.T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1,
                       extent=[x_coords[0]*1000, x_coords[-1]*1000, 0, lz*1000])
        for bx in bh_x:
            axes[1].axvline(bx*1000, color="red", linewidth=0.8, alpha=0.7)
        axes[1].set_title("Side view (max over Y) — red = bulkheads")
        axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Z (mm)")

        fig.suptitle("Complete Board: Shell + Bulkheads", fontsize=13)
        fig.tight_layout()
        out = os.path.join(output_dir, "complete_board_overview.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Overview: {out}")
    except ImportError:
        pass


def volume_to_stl(volume, x_coords, ly, lz, threshold: float, output_path: str):
    from skimage.measure import marching_cubes

    nx, nelz, nely = volume.shape
    dx = (x_coords[-1] - x_coords[0]) / max(nx - 1, 1)
    dy = ly / nely
    dz = lz / nelz

    print(f"Marching cubes (threshold={threshold})...")
    verts, faces, _, _ = marching_cubes(volume, level=threshold, spacing=(dx, dz, dy))
    # Remap axes: (X, Z, Y) → (X, Y, Z) and offset X
    verts_xyz = np.column_stack([
        verts[:, 0] + x_coords[0],
        verts[:, 2],
        verts[:, 1],
    ])
    print(f"Mesh: {len(verts_xyz):,} vertices, {len(faces):,} triangles")
    _write_stl(verts_xyz, faces, output_path)
    print(f"STL: {output_path}")
    bb = verts_xyz
    print(f"Bbox: X=[{bb[:,0].min()*1000:.0f},{bb[:,0].max()*1000:.0f}]mm  "
          f"Y=[{bb[:,1].min()*1000:.0f},{bb[:,1].max()*1000:.0f}]mm  "
          f"Z=[{bb[:,2].min()*1000:.0f},{bb[:,2].max()*1000:.0f}]mm")


def _write_stl(verts, faces, path):
    try:
        from stl import mesh as stl_mesh
        m = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            m.vectors[i] = verts[f]
        m.save(path)
    except ImportError:
        with open(path, "w") as f:
            f.write("solid board\n")
            for tri in faces:
                v = verts[tri]
                n = np.cross(v[1] - v[0], v[2] - v[0])
                nn = np.linalg.norm(n)
                if nn > 0:
                    n /= nn
                f.write(f"facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n  outer loop\n")
                for vv in v:
                    f.write(f"    vertex {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
                f.write("  endloop\nendfacet\n")
            f.write("endsolid board\n")


def main():
    parser = argparse.ArgumentParser(description="Build complete board: shell + bulkheads")
    parser.add_argument("--cross-sections", default="results/cross_sections_dense",
                        help="Phase 2 cross-sections directory")
    parser.add_argument("--output", default="results/complete_board/board.stl",
                        help="Output STL path")
    parser.add_argument("--n-bulkheads", type=int, default=10,
                        help="Number of evenly-spaced bulkhead plates (default 10)")
    parser.add_argument("--plate-thickness", type=float, default=4.0,
                        help="Bulkhead plate thickness in mm (default 4)")
    parser.add_argument("--dx", type=float, default=5.0,
                        help="X voxel resolution in mm (default 5)")
    parser.add_argument("--shell-frac", type=float, default=0.10,
                        help="Shell thickness as fraction of surface distance (default 0.10)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuously interpolated structure instead of discrete plates")
    args = parser.parse_args()

    from foilopt.geometry.board import FoilBoard, load_board_shape
    board = FoilBoard()
    board_shape = load_board_shape()
    if board_shape is None:
        raise RuntimeError("board_shape.s3dx not found — required for shell generation")

    sections, _ = load_cross_sections(args.cross_sections)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    volume, x_coords, ly, lz, bh_x = build_volume(
        board, board_shape, sections,
        n_bulkheads=args.n_bulkheads,
        plate_thickness_mm=args.plate_thickness,
        dx_mm=args.dx,
        shell_frac=args.shell_frac,
        continuous=args.continuous,
    )

    save_overview(volume, x_coords, ly, lz, bh_x, os.path.dirname(args.output))
    volume_to_stl(volume, x_coords, ly, lz, threshold=args.threshold, output_path=args.output)

    np.save(args.output.replace(".stl", "_volume.npy"), volume.astype(np.float32))
    print(f"\nDone. Open {args.output} in MeshLab, Blender, or your slicer.")


if __name__ == "__main__":
    main()
