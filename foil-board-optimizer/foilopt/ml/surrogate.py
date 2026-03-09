"""
Surrogate model for fast compliance prediction.

Instead of running full FEA for every candidate design, train a neural
network to predict compliance from the density field. This accelerates
the outer-loop exploration dramatically.
"""

import numpy as np
from typing import Optional, Tuple
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SurrogateModel:
    """Neural surrogate for FEA compliance prediction.

    Maps a flattened density vector to scalar compliance value.
    Trained on (density, compliance) pairs collected from FEA runs.

    The model uses a simple 3D CNN that takes the density field as a
    volumetric input and predicts compliance and max displacement.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        n_outputs: int = 2,
        hidden_channels: int = 32,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for surrogate model. pip install torch")

        self.grid_shape = grid_shape
        self.device = torch.device(device)
        self.model = self._build_model(grid_shape, n_outputs, hidden_channels)
        self.model.to(self.device)
        self.is_trained = False

        # Training data buffer
        self.X_buffer = []
        self.y_buffer = []

    def _build_model(
        self, grid_shape: tuple, n_outputs: int, hc: int
    ) -> "nn.Module":
        """Build a 3D CNN surrogate."""
        nx, ny, nz = grid_shape

        model = nn.Sequential(
            # Input: (batch, 1, nx, ny, nz)
            nn.Conv3d(1, hc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(hc, hc * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(hc * 2, hc * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hc * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
        )
        return model

    def add_sample(self, density: np.ndarray, targets: np.ndarray):
        """Add a training sample from an FEA evaluation.

        Args:
            density: Flat density vector.
            targets: [compliance, max_displacement] or similar.
        """
        self.X_buffer.append(density.copy())
        self.y_buffer.append(np.array(targets))

    def train(
        self, epochs: int = 100, lr: float = 1e-3, batch_size: int = 16
    ) -> dict:
        """Train the surrogate on collected samples.

        Returns:
            Dict with training loss history.
        """
        if len(self.X_buffer) < 5:
            return {"error": "Need at least 5 samples to train"}

        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)

        # Reshape to 3D grid
        nx, ny, nz = self.grid_shape
        X_3d = X.reshape(-1, 1, nx, ny, nz)

        # Normalize targets
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-8
        y_norm = (y - self.y_mean) / self.y_std

        X_t = torch.FloatTensor(X_3d).to(self.device)
        y_t = torch.FloatTensor(y_norm).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(loader))

        self.is_trained = True
        return {"loss_history": losses, "final_loss": losses[-1], "n_samples": len(X)}

    def predict(self, density: np.ndarray) -> np.ndarray:
        """Predict compliance from density field.

        Args:
            density: Flat density vector.

        Returns:
            Predicted [compliance, max_displacement].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        nx, ny, nz = self.grid_shape
        X = density.reshape(1, 1, nx, ny, nz)
        X_t = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(X_t).cpu().numpy()[0]

        return pred_norm * self.y_std + self.y_mean

    def save(self, path: str):
        """Save model and normalization params."""
        torch.save(self.model.state_dict(), path + ".pt")
        meta = {
            "grid_shape": self.grid_shape,
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
            "n_samples": len(self.X_buffer),
        }
        with open(path + ".json", "w") as f:
            json.dump(meta, f)

    def load(self, path: str):
        """Load a saved model."""
        self.model.load_state_dict(torch.load(path + ".pt", weights_only=True))
        with open(path + ".json") as f:
            meta = json.load(f)
        self.y_mean = np.array(meta["y_mean"])
        self.y_std = np.array(meta["y_std"])
        self.is_trained = True
