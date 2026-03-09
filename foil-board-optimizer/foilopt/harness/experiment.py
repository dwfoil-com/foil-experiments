"""
Experiment definition and tracking.

Each experiment represents one configuration of the optimization pipeline,
including mesh resolution, SIMP parameters, load cases, and ML settings.
The outer loop creates and evaluates experiments to converge on the
optimal board structure.
"""

import json
import time
import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Full configuration for one optimization experiment.

    The outer loop varies these parameters to explore the design space.
    """

    # Mesh resolution
    nelx: int = 28
    nely: int = 10
    nelz: int = 4

    # Board geometry
    board_length: float = 1.4
    board_width: float = 0.50
    board_thickness: float = 0.10

    # SIMP parameters
    volfrac: float = 0.35
    penal: float = 3.0
    rmin: float = 1.5
    max_iter: int = 100
    tol: float = 0.01
    use_heaviside: bool = False

    # Material (Young's modulus in Pa)
    E0: float = 2.0e9  # PETG
    nu: float = 0.35

    # Load case selection
    load_cases: list = field(default_factory=lambda: [
        "riding_normal", "pumping", "jump_landing", "carving"
    ])
    rider_weight_kg: float = 80.0

    # ML settings
    use_surrogate: bool = False
    use_neural_topo: bool = False
    surrogate_train_every: int = 10

    # Optimization method
    method: str = "simp"  # "simp", "neural_reparam", "evolutionary"

    def to_dict(self) -> dict:
        return asdict(self)

    def experiment_id(self) -> str:
        """Generate a unique ID from config hash."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class ExperimentResult:
    """Results from running one experiment."""

    experiment_id: str = ""
    config: Optional[ExperimentConfig] = None

    # Optimization results
    final_compliance: float = float("inf")
    final_volume_fraction: float = 0.0
    stiffness_score: float = 0.0
    max_displacement: float = float("inf")
    n_iterations: int = 0
    converged: bool = False
    optimization_time: float = 0.0

    # Per-load-case results
    load_case_results: dict = field(default_factory=dict)

    # Convergence data
    compliance_history: list = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    density_file: str = ""
    stl_file: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict() if self.config else {},
            "final_compliance": self.final_compliance,
            "final_volume_fraction": self.final_volume_fraction,
            "stiffness_score": self.stiffness_score,
            "max_displacement": self.max_displacement,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "optimization_time": self.optimization_time,
            "load_case_results": self.load_case_results,
            "timestamp": self.timestamp,
            "density_file": self.density_file,
            "stl_file": self.stl_file,
            "error": self.error,
        }
        return d


class Experiment:
    """Runs a single optimization experiment end-to-end.

    Given an ExperimentConfig, sets up the mesh, board, load cases,
    runs the optimization, evaluates results, and exports artifacts.
    """

    def __init__(self, config: ExperimentConfig, output_dir: str = "results"):
        self.config = config
        self.output_dir = output_dir
        self.exp_id = config.experiment_id()

    def run(self) -> ExperimentResult:
        """Execute the experiment."""
        from ..geometry.board import FoilBoard, create_default_load_cases
        from ..geometry.mesh import generate_hex_mesh
        from ..topology.simp import SIMPOptimizer, SIMPConfig
        from ..utils.export import export_density_to_stl

        result = ExperimentResult(
            experiment_id=self.exp_id,
            config=self.config,
            timestamp=datetime.now().isoformat(),
        )

        try:
            # Setup board
            board = FoilBoard(
                length=self.config.board_length,
                width=self.config.board_width,
                thickness=self.config.board_thickness,
            )

            # Generate mesh
            mesh = generate_hex_mesh(
                *board.get_domain_shape(),
                self.config.nelx,
                self.config.nely,
                self.config.nelz,
            )

            # Setup load cases
            all_cases = {lc.name: lc for lc in create_default_load_cases()}
            load_cases = [all_cases[name] for name in self.config.load_cases
                          if name in all_cases]

            if not load_cases:
                load_cases = create_default_load_cases()

            # Configure SIMP
            simp_config = SIMPConfig(
                volfrac=self.config.volfrac,
                penal=self.config.penal,
                rmin=self.config.rmin,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                use_heaviside=self.config.use_heaviside,
            )

            # Run optimization
            optimizer = SIMPOptimizer(mesh, board, simp_config)
            simp_result = optimizer.optimize(load_cases)

            # Store results
            result.final_compliance = simp_result.final_compliance
            result.final_volume_fraction = simp_result.final_volume
            result.n_iterations = simp_result.n_iterations
            result.converged = simp_result.converged
            result.optimization_time = simp_result.total_time
            result.compliance_history = simp_result.compliance_history

            if "aggregate" in simp_result.stiffness_metrics:
                agg = simp_result.stiffness_metrics["aggregate"]
                result.stiffness_score = agg["stiffness_score"]
                result.max_displacement = agg["max_displacement"]

            result.load_case_results = {
                k: v for k, v in simp_result.stiffness_metrics.items()
                if k != "aggregate"
            }

            # Save density field
            exp_dir = os.path.join(self.output_dir, self.exp_id)
            os.makedirs(exp_dir, exist_ok=True)

            density_path = os.path.join(exp_dir, "density.npy")
            import numpy as np
            np.save(density_path, simp_result.density)
            result.density_file = density_path

            # Export STL
            stl_path = os.path.join(exp_dir, "board_structure.stl")
            export_density_to_stl(
                simp_result.density, mesh, threshold=0.5, output_path=stl_path
            )
            result.stl_file = stl_path

            # Save result JSON
            with open(os.path.join(exp_dir, "result.json"), "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

        except Exception as e:
            result.error = str(e)
            import traceback
            result.error += "\n" + traceback.format_exc()

        return result
