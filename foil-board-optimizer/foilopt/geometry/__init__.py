"""Board geometry definitions and mesh generation."""

from .board import FoilBoard, LoadCase
from .mesh import generate_hex_mesh

__all__ = ["FoilBoard", "LoadCase", "generate_hex_mesh"]
