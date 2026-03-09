"""Run foil board topology optimization on Modal.

Usage:
    # First time: authenticate with Modal
    modal setup

    # Run optimization (defaults to 42x15x6, 100 iters)
    python modal_run.py

    # Higher resolution
    python modal_run.py --nelx 70 --nely 25 --nelz 10 --max-iter 200

    # Build viewer from results
    python build_viewer.py results/modal_latest
"""

import modal
import argparse
import os
import json
import sys

app = modal.App("foil-board-optimizer")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "numpy-stl>=3.0",
    )
    .add_local_dir(
        "foilopt",
        remote_path="/root/foilopt",
    )
    .add_local_dir(
        "configs",
        remote_path="/root/configs",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=3600,
)
def run_optimization(
    nelx: int = 42,
    nely: int = 15,
    nelz: int = 6,
    volfrac: float = 0.30,
    penal: float = 3.0,
    rmin: float = 1.5,
    max_iter: int = 100,
    tol: float = 0.005,
) -> dict:
    """Run SIMP topology optimization on Modal."""
    import numpy as np
    import time

    sys.path.insert(0, "/root")
    from foilopt.geometry.board import FoilBoard, create_default_load_cases
    from foilopt.geometry.mesh import generate_hex_mesh
    from foilopt.topology.simp import SIMPOptimizer, SIMPConfig
    from foilopt.utils.export import export_density_to_stl

    board = FoilBoard()
    mesh = generate_hex_mesh(*board.get_domain_shape(), nelx, nely, nelz)
    print(f"Mesh: {nelx}x{nely}x{nelz} = {mesh.n_elements} elements")
    print(f"Nodes: {mesh.n_nodes}, DOFs: {3 * mesh.n_nodes}")

    load_cases = create_default_load_cases()
    config = SIMPConfig(
        volfrac=volfrac,
        penal=penal,
        rmin=rmin,
        max_iter=max_iter,
        tol=tol,
    )

    def callback(it, c, v, ch, xp):
        if it % 10 == 0 or it < 5:
            print(f"  Iter {it:3d}: c={c:.4f} v={v:.3f} ch={ch:.4f}")

    t0 = time.time()
    optimizer = SIMPOptimizer(mesh, board, config, callback=callback)
    print(f"Setup: {time.time() - t0:.1f}s")

    result = optimizer.optimize(load_cases)
    print(f"Done: {result.n_iterations} iters, {result.total_time:.0f}s")
    print(f"Compliance: {result.final_compliance:.4f}, Vol: {result.final_volume:.3f}")

    # Export STL to bytes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".stl") as f:
        export_density_to_stl(result.density, mesh, output_path=f.name, threshold=0.3)
        stl_bytes = open(f.name, "rb").read()

    meta = {
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": board.length, "ly": board.width, "lz": board.thickness,
        "mast_bounds": list(board.get_mast_mount_bounds()),
        "foot_bounds": list(board.get_foot_zone_bounds()),
        "volfrac": result.final_volume,
        "compliance": result.final_compliance,
        "iterations": result.n_iterations,
        "time_seconds": result.total_time,
        "load_cases": [
            {"name": lc.name, "mast_force": lc.mast_force.tolist(), "mast_torque": lc.mast_torque.tolist()}
            for lc in load_cases
        ],
    }

    return {
        "density": result.density.astype(np.float32).tobytes(),
        "density_shape": list(result.density.shape),
        "stl": stl_bytes,
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser(description="Run foil board optimizer on Modal")
    parser.add_argument("--nelx", type=int, default=42)
    parser.add_argument("--nely", type=int, default=15)
    parser.add_argument("--nelz", type=int, default=6)
    parser.add_argument("--volfrac", type=float, default=0.30)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=1.5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.005)
    parser.add_argument("--output", default="results/modal_latest")
    args = parser.parse_args()

    print(f"Launching on Modal: {args.nelx}x{args.nely}x{args.nelz} mesh...")

    with modal.enable_output():
        with app.run():
            result = run_optimization.remote(
                nelx=args.nelx,
                nely=args.nely,
                nelz=args.nelz,
                volfrac=args.volfrac,
                penal=args.penal,
                rmin=args.rmin,
                max_iter=args.max_iter,
                tol=args.tol,
            )

    # Save results locally
    import numpy as np
    os.makedirs(args.output, exist_ok=True)

    density = np.frombuffer(result["density"], dtype=np.float32)
    np.save(os.path.join(args.output, "density.npy"), density)
    density.tofile(os.path.join(args.output, "density.bin"))

    with open(os.path.join(args.output, "board.stl"), "wb") as f:
        f.write(result["stl"])

    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(result["meta"], f, indent=2)

    meta = result["meta"]
    print(f"\nResults saved to {args.output}/")
    print(f"  Compliance: {meta['compliance']:.4f}")
    print(f"  Volume: {meta['volfrac']:.3f}")
    print(f"  Iterations: {meta['iterations']}")
    print(f"  Time: {meta['time_seconds']:.0f}s")
    print(f"\nBuild viewer: python build_viewer.py {args.output}")


if __name__ == "__main__":
    main()
