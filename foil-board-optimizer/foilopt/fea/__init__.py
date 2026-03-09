"""Finite Element Analysis for 3D linear elasticity."""

from .solver import FEASolver3D
from .element import hex8_stiffness_matrix

__all__ = ["FEASolver3D", "hex8_stiffness_matrix"]
