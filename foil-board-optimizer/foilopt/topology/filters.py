"""
Density filters and projections for topology optimization.

Filters prevent checkerboard patterns and ensure mesh-independent solutions.
Heaviside projection pushes intermediate densities toward 0/1.
"""

import numpy as np
from scipy import sparse
from ..geometry.mesh import HexMesh


def build_filter_matrix(mesh: HexMesh, rmin: float) -> sparse.csc_matrix:
    """Build a cone-weighted density filter matrix.

    Uses KDTree for efficient neighbor lookup instead of O(n^2) brute force.

    Args:
        mesh: The hex mesh.
        rmin: Filter radius (in physical units).

    Returns:
        Sparse filter matrix H (n_elements x n_elements), row-normalized.
    """
    from scipy.spatial import cKDTree

    centers = mesh.element_centers()
    n = mesh.n_elements

    tree = cKDTree(centers)
    pairs = tree.query_pairs(rmin, output_type='ndarray')

    # Build COO arrays: each pair (i,j) contributes both (i,j) and (j,i)
    if len(pairs) > 0:
        ii, jj = pairs[:, 0], pairs[:, 1]
        d_ij = np.linalg.norm(centers[ii] - centers[jj], axis=1)
        w_ij = rmin - d_ij

        # Both directions + diagonal
        rows = np.concatenate([ii, jj, np.arange(n)])
        cols = np.concatenate([jj, ii, np.arange(n)])
        vals = np.concatenate([w_ij, w_ij, np.full(n, rmin)])
    else:
        rows = np.arange(n)
        cols = np.arange(n)
        vals = np.full(n, rmin)

    H = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

    # Normalize rows
    row_sums = np.array(H.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    Hs = sparse.diags(1.0 / row_sums) @ H

    return Hs


def density_filter(
    density: np.ndarray, H: sparse.csc_matrix
) -> np.ndarray:
    """Apply density filter to prevent checkerboard patterns.

    Args:
        density: Raw element densities.
        H: Pre-computed filter matrix.

    Returns:
        Filtered densities.
    """
    return np.array(H @ density).flatten()


def heaviside_projection(
    density: np.ndarray, beta: float = 1.0, eta: float = 0.5
) -> np.ndarray:
    """Heaviside projection to push densities toward 0/1.

    Uses a smooth approximation of the Heaviside step function.
    Higher beta = sharper threshold.

    Args:
        density: Filtered element densities.
        beta: Sharpness parameter (increase during optimization).
        eta: Threshold (typically 0.5).

    Returns:
        Projected densities closer to 0 or 1.
    """
    num = np.tanh(beta * eta) + np.tanh(beta * (density - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den
