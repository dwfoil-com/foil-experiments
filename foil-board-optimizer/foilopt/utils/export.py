"""
Export optimized density fields to 3D-printable formats (STL).

Converts the voxel density field to a surface mesh by thresholding
and generating faces for exposed voxel surfaces (marching cubes or
simple box meshing).
"""

import numpy as np
from typing import Optional
import os


def density_to_voxels(
    density: np.ndarray, mesh, threshold: float = 0.5
) -> np.ndarray:
    """Convert flat density vector to 3D boolean voxel grid.

    Args:
        density: (N_elements,) density vector.
        mesh: HexMesh with nelx, nely, nelz.
        threshold: Density threshold for solid/void.

    Returns:
        (nelx, nely, nelz) boolean array.
    """
    voxels = density.reshape(mesh.nelz, mesh.nely, mesh.nelx).transpose(2, 1, 0)
    return voxels >= threshold


def voxels_to_stl_vertices_faces(
    voxels: np.ndarray, dx: float, dy: float, dz: float
) -> tuple:
    """Convert voxel grid to triangle mesh (vertices and faces).

    Generates faces only for exposed surfaces (where a solid voxel
    is adjacent to a void voxel or the domain boundary).

    Returns:
        (vertices, faces) - numpy arrays for STL export.
    """
    nx, ny, nz = voxels.shape
    vertices = []
    faces = []

    def add_quad(v0, v1, v2, v3):
        """Add a quad as two triangles."""
        base = len(vertices)
        vertices.extend([v0, v1, v2, v3])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not voxels[i, j, k]:
                    continue

                x0, y0, z0 = i * dx, j * dy, k * dz
                x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

                # Check each face for exposure
                # -X face
                if i == 0 or not voxels[i - 1, j, k]:
                    add_quad(
                        [x0, y0, z0], [x0, y1, z0],
                        [x0, y1, z1], [x0, y0, z1],
                    )
                # +X face
                if i == nx - 1 or not voxels[i + 1, j, k]:
                    add_quad(
                        [x1, y0, z0], [x1, y0, z1],
                        [x1, y1, z1], [x1, y1, z0],
                    )
                # -Y face
                if j == 0 or not voxels[i, j - 1, k]:
                    add_quad(
                        [x0, y0, z0], [x0, y0, z1],
                        [x1, y0, z1], [x1, y0, z0],
                    )
                # +Y face
                if j == ny - 1 or not voxels[i, j + 1, k]:
                    add_quad(
                        [x0, y1, z0], [x1, y1, z0],
                        [x1, y1, z1], [x0, y1, z1],
                    )
                # -Z face
                if k == 0 or not voxels[i, j, k - 1]:
                    add_quad(
                        [x0, y0, z0], [x1, y0, z0],
                        [x1, y1, z0], [x0, y1, z0],
                    )
                # +Z face
                if k == nz - 1 or not voxels[i, j, k + 1]:
                    add_quad(
                        [x0, y0, z1], [x0, y1, z1],
                        [x1, y1, z1], [x1, y0, z1],
                    )

    return np.array(vertices), np.array(faces)


def export_density_to_stl(
    density: np.ndarray,
    mesh,
    threshold: float = 0.5,
    output_path: str = "board_structure.stl",
):
    """Export density field to STL file for 3D printing.

    Args:
        density: Flat density vector from optimization.
        mesh: HexMesh defining the grid.
        threshold: Density cutoff for solid/void.
        output_path: Output STL file path.
    """
    voxels = density_to_voxels(density, mesh, threshold)
    vertices, faces = voxels_to_stl_vertices_faces(
        voxels, mesh.dx, mesh.dy, mesh.dz
    )

    if len(faces) == 0:
        print("Warning: No solid elements above threshold. Skipping STL export.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write binary STL
    n_triangles = len(faces)
    with open(output_path, "wb") as f:
        # Header (80 bytes)
        f.write(b"\0" * 80)
        # Number of triangles
        f.write(np.uint32(n_triangles).tobytes())

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len

            # Write: normal (3 floats), v0, v1, v2 (3 floats each), attribute (uint16)
            f.write(np.float32(normal).tobytes())
            f.write(np.float32(v0).tobytes())
            f.write(np.float32(v1).tobytes())
            f.write(np.float32(v2).tobytes())
            f.write(np.uint16(0).tobytes())

    print(f"Exported STL: {output_path} ({n_triangles} triangles)")
