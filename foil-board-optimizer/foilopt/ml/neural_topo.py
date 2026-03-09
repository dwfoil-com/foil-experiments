"""
Neural topology predictor.

Uses a neural network to directly predict optimal density fields,
bypassing the iterative SIMP loop. The network learns from SIMP
solutions and can generalize to new load cases or board geometries.

This is the "neural reparameterization" approach from DL4TO.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class UNet3D(nn.Module):
    """Simple 3D U-Net for density field prediction.

    Input: (batch, C_in, nx, ny, nz) - load/BC encoding
    Output: (batch, 1, nx, ny, nz) - predicted density field
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        bc = base_channels

        # Encoder
        self.enc1 = self._block(in_channels, bc)
        self.enc2 = self._block(bc, bc * 2)
        self.enc3 = self._block(bc * 2, bc * 4)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._block(bc * 4, bc * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(bc * 8, bc * 4)
        self.up2 = nn.ConvTranspose3d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(bc * 4, bc * 2)
        self.up1 = nn.ConvTranspose3d(bc * 2, bc, kernel_size=2, stride=2)
        self.dec1 = self._block(bc * 2, bc)

        self.out_conv = nn.Conv3d(bc, 1, kernel_size=1)

    def _block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        # Handle size mismatch from pooling
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Crop or pad x to match target spatial dimensions."""
        if x.shape[2:] != target.shape[2:]:
            x = nn.functional.interpolate(x, size=target.shape[2:], mode="trilinear", align_corners=False)
        return x


class NeuralTopologyPredictor:
    """Predicts optimal density fields using a trained 3D U-Net.

    The input encodes the load case as a volumetric field:
    - Channel 0: load magnitude at each node (interpolated to elements)
    - Channel 1: boundary condition mask (1 = fixed, 0 = free)
    - Channel 2: passive element mask (1 = must be solid)

    The output is the predicted density field.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. pip install torch")

        self.grid_shape = grid_shape
        self.device = torch.device(device)
        self.model = UNet3D(in_channels=3, base_channels=16)
        self.model.to(self.device)
        self.is_trained = False

    def encode_input(
        self,
        load_field: np.ndarray,
        bc_mask: np.ndarray,
        passive_mask: np.ndarray,
    ) -> np.ndarray:
        """Encode load case as 3-channel volumetric input.

        Args:
            load_field: (nx, ny, nz) magnitude of applied forces.
            bc_mask: (nx, ny, nz) 1 where BCs are fixed.
            passive_mask: (nx, ny, nz) 1 where elements must be solid.

        Returns:
            (3, nx, ny, nz) input tensor.
        """
        return np.stack([load_field, bc_mask, passive_mask], axis=0)

    def predict(self, encoded_input: np.ndarray) -> np.ndarray:
        """Predict density field from encoded input.

        Returns:
            (nx, ny, nz) predicted density field.
        """
        x = torch.FloatTensor(encoded_input).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        return pred.cpu().numpy()[0, 0]

    def train_on_simp_results(
        self,
        inputs: list,
        targets: list,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> dict:
        """Train on SIMP-optimized density fields.

        Args:
            inputs: List of (3, nx, ny, nz) encoded inputs.
            targets: List of (nx, ny, nz) SIMP density fields.

        Returns:
            Training metrics dict.
        """
        X = torch.FloatTensor(np.array(inputs)).to(self.device)
        y = torch.FloatTensor(np.array(targets)).unsqueeze(1).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Combined loss: MSE on density + volume constraint + binary penalty
        losses = []
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X)

            # Reconstruction loss
            mse_loss = nn.functional.mse_loss(pred, y)

            # Binary penalty (push toward 0/1)
            binary_loss = torch.mean(pred * (1 - pred))

            loss = mse_loss + 0.1 * binary_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.is_trained = True
        return {"loss_history": losses, "final_loss": losses[-1]}

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.is_trained = True
