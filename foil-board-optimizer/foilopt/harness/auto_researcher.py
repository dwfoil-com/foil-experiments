"""
Autonomous Outer-Loop Research Harness.

Inspired by Saab's aircraft auto-design methodology and the concept of an
AI-driven experimental loop. This harness autonomously:

1. Generates hypotheses about what optimization parameters will yield
   the stiffest board structure
2. Configures and runs experiments (inner-loop SIMP optimization)
3. Analyzes results to identify patterns and insights
4. Proposes new experiments based on what it learned
5. Tracks convergence toward the global optimum

The outer loop has full freedom to:
- Vary mesh resolution (coarse exploration → fine refinement)
- Adjust SIMP parameters (penalization, volume fraction, filter radius)
- Try different load case combinations and weights
- Switch between optimization methods (SIMP, evolutionary, neural)
- Modify board geometry parameters
- Install/use different packages if needed

This is designed to be driven by Claude Code autonomously in long-running
research sessions, where each session picks up from the best results so far.
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from .experiment import Experiment, ExperimentConfig, ExperimentResult


@dataclass
class ResearchState:
    """Persistent state of the research campaign.

    Saved between sessions so the outer loop can resume.
    """

    # All experiments run so far
    experiments: list = field(default_factory=list)

    # Best result found
    best_experiment_id: str = ""
    best_stiffness_score: float = 0.0
    best_compliance: float = float("inf")

    # Search state
    current_phase: str = "exploration"  # exploration, refinement, validation
    generation: int = 0
    total_fea_solves: int = 0
    total_time_hours: float = 0.0

    # Parameter ranges being explored
    param_ranges: dict = field(default_factory=lambda: {
        "volfrac": {"min": 0.15, "max": 0.60, "best": 0.35},
        "penal": {"min": 2.0, "max": 5.0, "best": 3.0},
        "rmin": {"min": 1.0, "max": 3.0, "best": 1.5},
        "nelx": {"min": 14, "max": 56, "best": 28},
        "nely": {"min": 5, "max": 20, "best": 10},
        "nelz": {"min": 2, "max": 8, "best": 4},
    })

    # Insights gathered
    insights: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiments": self.experiments,
            "best_experiment_id": self.best_experiment_id,
            "best_stiffness_score": self.best_stiffness_score,
            "best_compliance": self.best_compliance,
            "current_phase": self.current_phase,
            "generation": self.generation,
            "total_fea_solves": self.total_fea_solves,
            "total_time_hours": self.total_time_hours,
            "param_ranges": self.param_ranges,
            "insights": self.insights,
        }


class AutoResearcher:
    """Autonomous research driver for foil board optimization.

    This is the outer loop that designs experiments, runs them,
    and converges toward the optimal board structure.

    The research proceeds in phases:

    Phase 1: EXPLORATION (broad parameter sweep)
        - Coarse mesh, wide parameter ranges
        - Latin hypercube sampling of parameter space
        - Goal: identify promising regions

    Phase 2: REFINEMENT (focused optimization)
        - Increase mesh resolution around best configs
        - Narrow parameter ranges based on Phase 1 insights
        - Try Heaviside projection for sharper structures
        - Goal: converge on optimal parameters

    Phase 3: VALIDATION (high-fidelity confirmation)
        - Finest mesh resolution
        - Multiple load cases with realistic weights
        - Export final STL for 3D printing
        - Goal: production-ready design

    Usage:
        researcher = AutoResearcher(output_dir="results")
        researcher.run(max_experiments=50)
    """

    def __init__(
        self,
        output_dir: str = "results",
        state_file: str = "research_state.json",
    ):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, state_file)
        self.state = self._load_state()

    def _load_state(self) -> ResearchState:
        """Load or create research state."""
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                data = json.load(f)
            state = ResearchState()
            for k, v in data.items():
                if hasattr(state, k):
                    setattr(state, k, v)
            return state
        return ResearchState()

    def _save_state(self):
        """Persist research state."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)

    def _generate_exploration_configs(self, n: int = 8) -> list:
        """Generate diverse configs for exploration phase.

        Uses Latin Hypercube Sampling over the parameter space.
        """
        configs = []
        ranges = self.state.param_ranges

        # Latin Hypercube-like sampling
        for i in range(n):
            frac = (i + np.random.uniform(0, 1)) / n

            config = ExperimentConfig(
                # Coarse mesh for exploration
                nelx=int(np.interp(frac, [0, 1], [ranges["nelx"]["min"], ranges["nelx"]["max"] // 2])),
                nely=int(np.interp(frac, [0, 1], [ranges["nely"]["min"], ranges["nely"]["max"] // 2])),
                nelz=int(np.interp(frac, [0, 1], [ranges["nelz"]["min"], ranges["nelz"]["max"] // 2])),
                volfrac=np.interp(
                    np.random.uniform(), [0, 1],
                    [ranges["volfrac"]["min"], ranges["volfrac"]["max"]],
                ),
                penal=np.interp(
                    np.random.uniform(), [0, 1],
                    [ranges["penal"]["min"], ranges["penal"]["max"]],
                ),
                rmin=np.interp(
                    np.random.uniform(), [0, 1],
                    [ranges["rmin"]["min"], ranges["rmin"]["max"]],
                ),
                max_iter=60,  # fewer iterations for exploration
            )
            configs.append(config)

        return configs

    def _generate_refinement_configs(self, n: int = 6) -> list:
        """Generate configs focused around the best-performing region.

        Increases mesh resolution and narrows parameter ranges.
        """
        configs = []
        ranges = self.state.param_ranges

        for i in range(n):
            # Perturb around the best values
            noise = lambda lo, hi, best: np.clip(
                best + np.random.normal(0, (hi - lo) * 0.15),
                lo, hi
            )

            config = ExperimentConfig(
                # Higher resolution mesh
                nelx=int(noise(ranges["nelx"]["min"], ranges["nelx"]["max"], ranges["nelx"]["best"])),
                nely=int(noise(ranges["nely"]["min"], ranges["nely"]["max"], ranges["nely"]["best"])),
                nelz=int(noise(ranges["nelz"]["min"], ranges["nelz"]["max"], ranges["nelz"]["best"])),
                volfrac=noise(ranges["volfrac"]["min"], ranges["volfrac"]["max"], ranges["volfrac"]["best"]),
                penal=noise(ranges["penal"]["min"], ranges["penal"]["max"], ranges["penal"]["best"]),
                rmin=noise(ranges["rmin"]["min"], ranges["rmin"]["max"], ranges["rmin"]["best"]),
                use_heaviside=np.random.random() > 0.5,
                max_iter=150,
            )
            configs.append(config)

        return configs

    def _generate_validation_configs(self, n: int = 3) -> list:
        """Generate high-fidelity validation configs.

        Uses the best parameters with finest mesh resolution.
        """
        configs = []
        ranges = self.state.param_ranges

        for i in range(n):
            config = ExperimentConfig(
                nelx=ranges["nelx"]["max"],
                nely=ranges["nely"]["max"],
                nelz=ranges["nelz"]["max"],
                volfrac=ranges["volfrac"]["best"],
                penal=ranges["penal"]["best"],
                rmin=ranges["rmin"]["best"],
                use_heaviside=True,
                max_iter=300,
                tol=0.005,
            )
            configs.append(config)

        return configs

    def _analyze_results(self, results: list):
        """Analyze experiment results and update state.

        Identifies patterns, updates parameter ranges, and
        generates insights for the next generation.
        """
        if not results:
            return

        # Find best result
        valid_results = [r for r in results if r.error is None
                         and r.final_compliance > 0 and np.isfinite(r.final_compliance)]
        if not valid_results:
            self.state.insights.append(
                f"Gen {self.state.generation}: All {len(results)} experiments failed or had invalid compliance"
            )
            return

        best = min(valid_results, key=lambda r: r.final_compliance)

        if best.final_compliance < self.state.best_compliance:
            self.state.best_stiffness_score = best.stiffness_score
            self.state.best_compliance = best.final_compliance
            self.state.best_experiment_id = best.experiment_id

            # Update parameter ranges toward best
            if best.config:
                c = best.config
                ranges = self.state.param_ranges
                for param in ["volfrac", "penal", "rmin"]:
                    val = getattr(c, param)
                    ranges[param]["best"] = val
                    # Narrow range around best
                    span = ranges[param]["max"] - ranges[param]["min"]
                    ranges[param]["min"] = max(ranges[param]["min"], val - span * 0.3)
                    ranges[param]["max"] = min(ranges[param]["max"], val + span * 0.3)

                for param in ["nelx", "nely", "nelz"]:
                    ranges[param]["best"] = getattr(c, param)

        # Generate insights
        compliances = [r.final_compliance for r in valid_results]
        volumes = [r.final_volume_fraction for r in valid_results]

        insight = (
            f"Gen {self.state.generation}: "
            f"{len(valid_results)}/{len(results)} succeeded. "
            f"Compliance range: [{min(compliances):.4f}, {max(compliances):.4f}]. "
            f"Volume range: [{min(volumes):.3f}, {max(volumes):.3f}]. "
            f"Best stiffness score: {best.stiffness_score:.6f}"
        )
        self.state.insights.append(insight)
        print(f"\n{'='*60}")
        print(f"INSIGHT: {insight}")
        print(f"{'='*60}\n")

    def _should_advance_phase(self) -> bool:
        """Decide if we should move to the next phase."""
        n_experiments = len(self.state.experiments)

        if self.state.current_phase == "exploration" and n_experiments >= 8:
            return True
        if self.state.current_phase == "refinement" and n_experiments >= 20:
            return True
        return False

    def _advance_phase(self):
        """Move to the next research phase."""
        phases = ["exploration", "refinement", "validation"]
        current_idx = phases.index(self.state.current_phase)
        if current_idx < len(phases) - 1:
            self.state.current_phase = phases[current_idx + 1]
            self.state.insights.append(
                f"Advanced to phase: {self.state.current_phase}"
            )
            print(f"\n*** ADVANCING TO PHASE: {self.state.current_phase.upper()} ***\n")

    def run(self, max_experiments: int = 50, experiments_per_gen: int = 4):
        """Run the autonomous research loop.

        Args:
            max_experiments: Stop after this many total experiments.
            experiments_per_gen: Experiments to run per generation.
        """
        print("=" * 60)
        print("FOIL BOARD STRUCTURE OPTIMIZER - AUTO RESEARCHER")
        print("=" * 60)
        print(f"Phase: {self.state.current_phase}")
        print(f"Experiments so far: {len(self.state.experiments)}")
        print(f"Best stiffness score: {self.state.best_stiffness_score:.6f}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)

        while len(self.state.experiments) < max_experiments:
            gen_start = time.time()
            self.state.generation += 1

            print(f"\n--- Generation {self.state.generation} "
                  f"(Phase: {self.state.current_phase}) ---")

            # Generate experiment configs based on current phase
            if self.state.current_phase == "exploration":
                configs = self._generate_exploration_configs(experiments_per_gen)
            elif self.state.current_phase == "refinement":
                configs = self._generate_refinement_configs(experiments_per_gen)
            else:
                configs = self._generate_validation_configs(
                    min(experiments_per_gen, 3)
                )

            # Run experiments
            gen_results = []
            for i, config in enumerate(configs):
                exp_num = len(self.state.experiments) + 1
                print(f"\n  Experiment {exp_num}/{max_experiments} "
                      f"(gen {self.state.generation}, #{i+1})")
                print(f"    mesh: {config.nelx}x{config.nely}x{config.nelz}, "
                      f"vf={config.volfrac:.2f}, p={config.penal:.1f}")

                experiment = Experiment(config, self.output_dir)
                result = experiment.run()

                if result.error:
                    print(f"    FAILED: {result.error[:100]}")
                else:
                    print(f"    compliance={result.final_compliance:.4f}, "
                          f"stiffness={result.stiffness_score:.6f}, "
                          f"vol={result.final_volume_fraction:.3f}, "
                          f"time={result.optimization_time:.1f}s")

                gen_results.append(result)
                self.state.experiments.append(result.to_dict())

                if len(self.state.experiments) >= max_experiments:
                    break

            # Analyze and learn
            self._analyze_results(gen_results)

            gen_time = time.time() - gen_start
            self.state.total_time_hours += gen_time / 3600

            # Check if we should advance phase
            if self._should_advance_phase():
                self._advance_phase()

            # Save state
            self._save_state()

            print(f"\n  Generation time: {gen_time:.1f}s")
            print(f"  Total experiments: {len(self.state.experiments)}")
            print(f"  Best compliance: {self.state.best_compliance:.4f}")

        # Final summary
        self._print_summary()

    def _print_summary(self):
        """Print final research summary."""
        print("\n" + "=" * 60)
        print("RESEARCH COMPLETE")
        print("=" * 60)
        print(f"Total experiments: {len(self.state.experiments)}")
        print(f"Total time: {self.state.total_time_hours:.2f} hours")
        print(f"Best experiment: {self.state.best_experiment_id}")
        print(f"Best stiffness score: {self.state.best_stiffness_score:.6f}")
        print(f"Best compliance: {self.state.best_compliance:.4f}")
        print(f"\nInsights:")
        for insight in self.state.insights:
            print(f"  - {insight}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"State saved to: {self.state_file}")

    def get_best_density(self) -> Optional[np.ndarray]:
        """Load the density field from the best experiment."""
        if not self.state.best_experiment_id:
            return None
        path = os.path.join(
            self.output_dir, self.state.best_experiment_id, "density.npy"
        )
        if os.path.exists(path):
            return np.load(path)
        return None
