"""Machine learning modules for topology optimization."""

from .surrogate import SurrogateModel
from .neural_topo import NeuralTopologyPredictor

__all__ = ["SurrogateModel", "NeuralTopologyPredictor"]
