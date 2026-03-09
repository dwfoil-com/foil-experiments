"""
Visualization utilities for topology optimization results.

Generates 3D voxel plots, convergence charts, and cross-section views
of the optimized board structure.
"""

import numpy as np
from typing import Optional
import os


def plot_convergence(result, save_path: Optional[str] = None):
    """Plot compliance and volume convergence history.

    Args:
        result: SIMPResult or ExperimentResult with history data.
        save_path: If set, save figure to file instead of showing.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    iterations = range(len(result.compliance_history))

    ax1.semilogy(iterations, result.compliance_history, "b-", linewidth=1.5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Compliance (lower = stiffer)")
    ax1.set_title("Compliance Convergence")
    ax1.grid(True, alpha=0.3)

    if hasattr(result, "volume_history") and result.volume_history:
        ax2.plot(iterations, result.volume_history, "r-", linewidth=1.5)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Volume Fraction")
        ax2.set_title("Volume Convergence")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_density_slices(
    density: np.ndarray,
    mesh,
    n_slices: int = 4,
    save_path: Optional[str] = None,
):
    """Plot cross-section slices through the board thickness.

    Args:
        density: Flat density vector.
        mesh: HexMesh.
        n_slices: Number of Z-slices to show.
        save_path: If set, save figure.
    """
    import matplotlib.pyplot as plt

    voxels = density.reshape(mesh.nelz, mesh.nely, mesh.nelx).transpose(2, 1, 0)

    slice_indices = np.linspace(0, mesh.nelz - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for ax, zi in zip(axes, slice_indices):
        im = ax.imshow(
            voxels[:, :, zi].T,
            cmap="YlOrRd",
            origin="lower",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        z_pos = zi * mesh.dz * 1000  # mm
        ax.set_title(f"Z = {z_pos:.1f} mm")
        ax.set_xlabel("X (elements)")
        ax.set_ylabel("Y (elements)")
        plt.colorbar(im, ax=ax, label="Density")

    plt.suptitle("Board Cross-Sections (Bottom → Deck)", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_3d_density(
    density: np.ndarray,
    mesh,
    threshold: float = 0.3,
    save_path: Optional[str] = None,
):
    """Plot 3D voxel visualization of the optimized structure.

    Args:
        density: Flat density vector.
        mesh: HexMesh.
        threshold: Minimum density to display.
        save_path: If set, save figure.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    voxels = density.reshape(mesh.nelz, mesh.nely, mesh.nelx).transpose(2, 1, 0)

    solid = voxels >= threshold

    # Build RGBA facecolors: density maps to alpha
    colors = np.zeros(voxels.shape + (4,))
    alphas = np.clip(voxels[solid], 0.3, 1.0)
    colors[solid, 0] = 0.8
    colors[solid, 1] = 0.2
    colors[solid, 2] = 0.1
    colors[solid, 3] = alphas
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(solid, facecolors=colors, edgecolors="gray", linewidth=0.2)

    ax.set_xlabel(f"X (length, {mesh.nelx} el)")
    ax.set_ylabel(f"Y (width, {mesh.nely} el)")
    ax.set_zlabel(f"Z (thickness, {mesh.nelz} el)")
    ax.set_title("Optimized Board Internal Structure")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
