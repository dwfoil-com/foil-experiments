"""
Foil board geometry definition.

The board is modeled as a 3D rectangular domain (simplified from the actual
curved shape) with key zones:
- Deck surface (top) where the rider stands and applies load
- Mast mount region (bottom center) where the foil mast bolts through
- Rails (sides) which provide structural support

Coordinate system:
  X = length (nose to tail)
  Y = width (rail to rail)
  Z = thickness (bottom to deck)

All dimensions in meters.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LoadCase:
    """A load scenario applied to the board.

    Attributes:
        name: Human-readable name for this load case.
        deck_pressure: Pressure distribution on deck (Pa) as (N_nodes,) array,
            or a scalar for uniform pressure.
        mast_force: Force vector [Fx, Fy, Fz] at mast mount (N).
        mast_torque: Torque vector [Tx, Ty, Tz] at mast mount (N·m).
        weight_rider_kg: Rider weight used to derive deck_pressure if not set.
    """

    name: str = "standing"
    deck_pressure: Optional[np.ndarray] = None
    mast_force: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -2000.0]))
    mast_torque: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    weight_rider_kg: float = 80.0

    def get_deck_force_total(self) -> float:
        """Total downward force from rider (N)."""
        return self.weight_rider_kg * 9.81


@dataclass
class FoilBoard:
    """Simplified foil board geometry.

    Attributes:
        length: Board length along X (m). Typical: 1.2 - 1.6m.
        width: Board width along Y (m). Typical: 0.45 - 0.55m.
        thickness: Board thickness along Z (m). Typical: 0.08 - 0.12m.
        mast_mount_x: X-position of mast mount center (fraction of length).
        mast_mount_y: Y-position of mast mount center (fraction of width).
        mast_mount_length: Length of mast mounting plate (m).
        mast_mount_width: Width of mast mounting plate (m).
        foot_zone_x: X-range for foot placement (fraction of length) [front, back].
        foot_zone_y: Y-range for foot placement (fraction of width) [left, right].
    """

    length: float = 1.4
    width: float = 0.50
    thickness: float = 0.10

    # Mast mount position (fractions of board dimensions)
    mast_mount_x: float = 0.45  # slightly forward of center
    mast_mount_y: float = 0.50  # centered
    mast_mount_length: float = 0.20  # ~20cm plate
    mast_mount_width: float = 0.12  # ~12cm plate

    # Foot zones (fractions)
    foot_zone_x: tuple = (0.30, 0.70)
    foot_zone_y: tuple = (0.25, 0.75)

    def get_mast_mount_bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) of the mast mount region in meters."""
        cx = self.mast_mount_x * self.length
        cy = self.mast_mount_y * self.width
        hl = self.mast_mount_length / 2
        hw = self.mast_mount_width / 2
        return (cx - hl, cx + hl, cy - hw, cy + hw)

    def get_foot_zone_bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) of the foot zone in meters."""
        return (
            self.foot_zone_x[0] * self.length,
            self.foot_zone_x[1] * self.length,
            self.foot_zone_y[0] * self.width,
            self.foot_zone_y[1] * self.width,
        )

    def get_domain_shape(self) -> tuple:
        """Return (Lx, Ly, Lz) domain dimensions."""
        return (self.length, self.width, self.thickness)

    def is_in_mast_mount(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for points inside the mast mount region."""
        xmin, xmax, ymin, ymax = self.get_mast_mount_bounds()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def is_in_foot_zone(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for points inside the foot zone."""
        xmin, xmax, ymin, ymax = self.get_foot_zone_bounds()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def create_default_load_cases() -> list:
    """Create standard load cases for foil board optimization.

    Returns:
        List of LoadCase objects covering typical riding scenarios.
    """
    cases = [
        # Normal riding - rider standing, mast pulling down and forward
        LoadCase(
            name="riding_normal",
            weight_rider_kg=80.0,
            mast_force=np.array([500.0, 0.0, -2000.0]),
            mast_torque=np.array([0.0, 200.0, 0.0]),
        ),
        # Pumping - dynamic loading, higher forces
        LoadCase(
            name="pumping",
            weight_rider_kg=80.0,
            mast_force=np.array([300.0, 0.0, -3000.0]),
            mast_torque=np.array([0.0, 400.0, 0.0]),
        ),
        # Landing from jump - impact load
        LoadCase(
            name="jump_landing",
            weight_rider_kg=80.0,
            mast_force=np.array([0.0, 0.0, -5000.0]),
            mast_torque=np.array([0.0, 100.0, 50.0]),
        ),
        # Carving turn - lateral loading
        LoadCase(
            name="carving",
            weight_rider_kg=80.0,
            mast_force=np.array([200.0, 800.0, -2500.0]),
            mast_torque=np.array([150.0, 100.0, 300.0]),
        ),
    ]
    return cases
