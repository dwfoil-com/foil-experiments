"""
Structured hexahedral mesh generation for the board domain.

Generates a regular 3D grid of hexahedral (brick) elements, with node
coordinates and element connectivity. The mesh is the foundation for both
FEA and topology optimization (one design variable per element).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class HexMesh:
    """A structured hexahedral mesh.

    Attributes:
        nodes: (N_nodes, 3) array of node coordinates.
        elements: (N_elements, 8) array of node indices per element.
        nelx: Number of elements along X.
        nely: Number of elements along Y.
        nelz: Number of elements along Z.
        dx, dy, dz: Element sizes.
    """

    nodes: np.ndarray
    elements: np.ndarray
    nelx: int
    nely: int
    nelz: int
    dx: float
    dy: float
    dz: float

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def element_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def element_centers(self) -> np.ndarray:
        """Return (N_elements, 3) array of element centroid coordinates."""
        return self.nodes[self.elements].mean(axis=1)

    def get_node_grid_index(self, i: int, j: int, k: int) -> int:
        """Convert (i, j, k) grid index to flat node index."""
        return i * (self.nely + 1) * (self.nelz + 1) + j * (self.nelz + 1) + k

    def density_to_3d(self, density: np.ndarray) -> np.ndarray:
        """Reshape a flat density vector to 3D grid (nelx, nely, nelz)."""
        return density.reshape(self.nelz, self.nely, self.nelx).transpose(2, 1, 0)


def generate_hex_mesh(
    lx: float, ly: float, lz: float, nelx: int, nely: int, nelz: int
) -> HexMesh:
    """Generate a structured hexahedral mesh over a rectangular domain.

    Args:
        lx, ly, lz: Domain dimensions.
        nelx, nely, nelz: Number of elements in each direction.

    Returns:
        HexMesh with nodes and element connectivity.
    """
    dx = lx / nelx
    dy = ly / nely
    dz = lz / nelz

    # Generate node coordinates
    x = np.linspace(0, lx, nelx + 1)
    y = np.linspace(0, ly, nely + 1)
    z = np.linspace(0, lz, nelz + 1)

    # Create 3D grid of nodes
    # With indexing="ij", shape is (nelx+1, nely+1, nelz+1).
    # C-order ravel: last axis (z) varies fastest, then y, then x.
    # So flat index = i*(nely+1)*(nelz+1) + j*(nelz+1) + k
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])

    # Generate element connectivity (8-node hexahedra)
    nyz = (nely + 1) * (nelz + 1)  # stride for i
    nz = nelz + 1                   # stride for j
    elements = np.zeros((nelx * nely * nelz, 8), dtype=np.int64)

    idx = 0
    for k in range(nelz):
        for j in range(nely):
            for i in range(nelx):
                n0 = i * nyz + j * nz + k
                n1 = (i + 1) * nyz + j * nz + k
                n2 = (i + 1) * nyz + (j + 1) * nz + k
                n3 = i * nyz + (j + 1) * nz + k
                n4 = n0 + 1      # same (i,j) but k+1
                n5 = n1 + 1
                n6 = n2 + 1
                n7 = n3 + 1
                # Bottom face CCW (z=k): n0,n1,n2,n3
                # Top face CCW (z=k+1): n4,n5,n6,n7
                elements[idx] = [n0, n1, n2, n3, n4, n5, n6, n7]
                idx += 1

    return HexMesh(
        nodes=nodes,
        elements=elements,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        dx=dx,
        dy=dy,
        dz=dz,
    )
