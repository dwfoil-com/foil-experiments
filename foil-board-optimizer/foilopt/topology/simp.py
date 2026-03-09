"""
SIMP (Solid Isotropic Material with Penalization) topology optimizer.

This is the core inner-loop optimizer that iteratively updates element
densities to minimize compliance (maximize stiffness) subject to a
volume constraint.

The optimization loop:
1. Solve FEA with current densities
2. Compute sensitivities (gradient of compliance w.r.t. density)
3. Filter sensitivities
4. Update densities using optimality criteria (OC)
5. Repeat until convergence
"""

import numpy as np
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

from ..geometry.mesh import HexMesh
from ..geometry.board import FoilBoard, LoadCase
from ..fea.solver import FEASolver3D
from .filters import build_filter_matrix, density_filter, heaviside_projection


@dataclass
class SIMPConfig:
    """Configuration for SIMP optimizer.

    Attributes:
        volfrac: Target volume fraction (0-1). Lower = lighter board.
        penal: SIMP penalization power. Higher pushes toward 0/1.
        rmin: Filter radius as multiple of element size.
        max_iter: Maximum optimization iterations.
        tol: Convergence tolerance on density change.
        use_heaviside: Whether to apply Heaviside projection.
        beta_init: Initial Heaviside sharpness.
        beta_max: Maximum Heaviside sharpness.
        beta_step: Multiply beta by this every N iterations.
        move_limit: Maximum density change per iteration (OC).
    """

    volfrac: float = 0.35
    penal: float = 3.0
    rmin: float = 1.5  # multiplied by max element size
    max_iter: int = 100
    tol: float = 0.01
    use_heaviside: bool = False
    beta_init: float = 1.0
    beta_max: float = 32.0
    beta_step: float = 2.0
    move_limit: float = 0.2


@dataclass
class SIMPResult:
    """Result from SIMP optimization.

    Attributes:
        density: Final optimized density field.
        compliance_history: Compliance at each iteration.
        volume_history: Volume fraction at each iteration.
        convergence_history: Max density change at each iteration.
        final_compliance: Last compliance value.
        final_volume: Last volume fraction.
        n_iterations: Number of iterations performed.
        converged: Whether the optimizer converged within tolerance.
        total_time: Total wall-clock time (seconds).
        stiffness_metrics: Per-load-case stiffness evaluation.
    """

    density: np.ndarray = field(default_factory=lambda: np.array([]))
    compliance_history: list = field(default_factory=list)
    volume_history: list = field(default_factory=list)
    convergence_history: list = field(default_factory=list)
    final_compliance: float = 0.0
    final_volume: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    total_time: float = 0.0
    stiffness_metrics: dict = field(default_factory=dict)


class SIMPOptimizer:
    """SIMP topology optimization for foil board.

    Usage:
        optimizer = SIMPOptimizer(mesh, board, config)
        result = optimizer.optimize(load_cases)
    """

    def __init__(
        self,
        mesh: HexMesh,
        board: FoilBoard,
        config: Optional[SIMPConfig] = None,
        callback: Optional[Callable] = None,
    ):
        self.mesh = mesh
        self.board = board
        self.config = config or SIMPConfig()
        self.callback = callback

        self.solver = FEASolver3D(
            mesh, board, penal=self.config.penal
        )

        # Build density filter
        rmin_phys = self.config.rmin * max(mesh.dx, mesh.dy, mesh.dz)
        self.H = build_filter_matrix(mesh, rmin_phys)

    def _init_density(self) -> np.ndarray:
        """Initialize density field.

        Start with uniform density = volume fraction.
        Force density=1 at mast mount and skin (top/bottom faces).
        """
        x = np.full(self.mesh.n_elements, self.config.volfrac)

        centers = self.mesh.element_centers()

        # Force solid at mast mount region
        mast_mask = self.board.is_in_mast_mount(centers[:, 0], centers[:, 1])
        x[mast_mask] = 1.0

        # Force solid at top and bottom skins (thin shell)
        skin_thickness = self.mesh.dz * 0.5
        top_mask = centers[:, 2] > (self.board.thickness - skin_thickness)
        bot_mask = centers[:, 2] < skin_thickness
        x[top_mask | bot_mask] = 1.0

        return x

    def _get_passive_elements(self) -> tuple:
        """Identify elements that must remain solid or void.

        Returns:
            (solid_mask, void_mask) boolean arrays.
        """
        centers = self.mesh.element_centers()

        # Must be solid: mast mount through-thickness, top/bottom skins
        mast_solid = self.board.is_in_mast_mount(centers[:, 0], centers[:, 1])
        skin_thickness = self.mesh.dz * 0.5
        top_skin = centers[:, 2] > (self.board.thickness - skin_thickness)
        bot_skin = centers[:, 2] < skin_thickness

        solid_mask = mast_solid | top_skin | bot_skin
        void_mask = np.zeros(self.mesh.n_elements, dtype=bool)

        return solid_mask, void_mask

    def _oc_update(
        self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray,
        solid_mask: np.ndarray, void_mask: np.ndarray,
    ) -> np.ndarray:
        """Optimality Criteria (OC) update for density.

        Bisection on Lagrange multiplier to satisfy volume constraint.
        Only free elements (not passive solid/void) participate in the
        volume constraint.
        """
        volfrac = self.config.volfrac
        move = self.config.move_limit
        n_elem = len(x)

        # Compute adjusted volume target for free elements only
        free_mask = ~solid_mask & ~void_mask
        n_total = n_elem
        n_solid = np.sum(solid_mask)
        n_free = np.sum(free_mask)

        if n_free == 0:
            xnew = x.copy()
            xnew[solid_mask] = 1.0
            xnew[void_mask] = 0.001
            return xnew

        # Target volume for free elements: total target minus forced-solid volume
        target_free_vol = (volfrac * n_total - n_solid) / n_free
        target_free_vol = np.clip(target_free_vol, 0.001, 1.0)

        l1, l2 = 0.0, 1e9

        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)

            # OC update rule
            ratio = np.maximum(-dc / (dv * lmid + 1e-12), 1e-12)
            Be = np.sqrt(ratio)
            xnew = np.maximum(0.001, np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * Be)),
            ))

            # Enforce passive elements
            xnew[solid_mask] = 1.0
            xnew[void_mask] = 0.001

            # Volume check on free elements only
            if np.mean(xnew[free_mask]) > target_free_vol:
                l1 = lmid
            else:
                l2 = lmid

        return xnew

    def optimize(self, load_cases: list) -> SIMPResult:
        """Run the SIMP optimization loop.

        Args:
            load_cases: List of LoadCase objects to optimize against.

        Returns:
            SIMPResult with optimized density and history.
        """
        t_start = time.time()
        config = self.config
        result = SIMPResult()

        x = self._init_density()
        solid_mask, void_mask = self._get_passive_elements()

        beta = config.beta_init if config.use_heaviside else 1.0

        for iteration in range(config.max_iter):
            # Apply filter
            xPhys = density_filter(x, self.H)
            if config.use_heaviside:
                xPhys = heaviside_projection(xPhys, beta)

            # Multi-load case: sum compliance and sensitivities
            total_compliance = 0.0
            dc = np.zeros(self.mesh.n_elements)

            for lc in load_cases:
                u, info = self.solver.solve(xPhys, lc)
                ce = self.solver.compute_element_compliance(xPhys, u)

                # Sensitivity of compliance w.r.t. density
                dc_lc = -config.penal * xPhys ** (config.penal - 1) * (
                    self.solver.E0 - self.solver.Emin
                ) * ce

                total_compliance += info["compliance"]
                dc += dc_lc

            # Chain rule for density filter: dc/dx = H^T @ dc/dxPhys
            dc = np.array(self.H.T @ dc).flatten()

            # Volume sensitivity
            dv = np.ones(self.mesh.n_elements)

            # OC update
            xnew = self._oc_update(x, dc, dv, solid_mask, void_mask)

            # Check convergence
            change = np.max(np.abs(xnew - x))
            vol = np.mean(xPhys)

            result.compliance_history.append(total_compliance)
            result.volume_history.append(vol)
            result.convergence_history.append(change)

            if self.callback:
                self.callback(iteration, total_compliance, vol, change, xPhys)

            x = xnew

            # Increase Heaviside sharpness periodically
            if config.use_heaviside and iteration % 20 == 0 and iteration > 0:
                beta = min(beta * config.beta_step, config.beta_max)

            if change < config.tol and iteration > 10:
                result.converged = True
                break

        # Final evaluation
        xPhys = density_filter(x, self.H)
        if config.use_heaviside:
            xPhys = heaviside_projection(xPhys, beta)

        result.density = xPhys
        result.final_compliance = result.compliance_history[-1]
        result.final_volume = result.volume_history[-1]
        result.n_iterations = len(result.compliance_history)
        result.total_time = time.time() - t_start

        # Compute stiffness metrics across all load cases
        result.stiffness_metrics = self.solver.compute_stiffness_metric(
            xPhys, load_cases
        )

        return result
