"""CNN Dueling DQN — spatial grid input (3 channels × H × W)."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Dueling CNN-DQN for Snake.

    Input  : (batch, channels * grid_height * grid_width) — flat grid, reshaped internally.
    Output : (batch, output_size)                         — Q-value per action.

    Architecture
    ────────────
    Three convolutional layers extract spatial features from the grid:
      • Conv1  (3 → 32, 3×3, same padding)   — detects local patterns
      • Conv2  (32 → 64, 3×3, same padding)  — regional patterns
      • Conv3  (64 → 64, 3×3, stride 2)      — spatial compression / translation invariance

    Flattened features → shared FC trunk (512) → Dueling heads:
      • Value stream     V(s)    — how good is this state?
      • Advantage stream A(s,a)  — how much better is action a vs average?

    Combined:  Q(s, a) = V(s) + A(s, a) − mean_{a'}[ A(s, a') ]
    """

    def __init__(
        self,
        input_size:  int = 2250,    # channels * grid_height * grid_width
        hidden_size: int = 512,
        output_size: int = 4,
        grid_height: int = 25,
        grid_width:  int = 30,
        channels:    int = 3,
    ) -> None:
        super().__init__()

        self.grid_height = grid_height
        self.grid_width  = grid_width
        self.channels    = channels

        # ── Convolutional feature extractor ──────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        # Output spatial size after stride-2 conv (no padding):
        #   out = floor((in - 3) / 2) + 1
        conv_h   = (grid_height - 3) // 2 + 1   # 25 → 12
        conv_w   = (grid_width  - 3) // 2 + 1   # 30 → 14
        conv_flat = 64 * conv_h * conv_w          # 64 × 12 × 14 = 10 752

        # ── Shared FC trunk ───────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(conv_flat, hidden_size),
            nn.ReLU(inplace=True),
        )

        # ── Dueling streams ───────────────────────────────────────────────
        stream_size = hidden_size // 2            # 256

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, stream_size),
            nn.ReLU(inplace=True),
            nn.Linear(stream_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, stream_size),
            nn.ReLU(inplace=True),
            nn.Linear(stream_size, output_size),
        )

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels*H*W) flat tensor — reshaped internally to (batch, C, H, W).
        Returns:
            Q-values: (batch, output_size)
        """
        # Reshape flat input to spatial grid
        x = x.view(-1, self.channels, self.grid_height, self.grid_width)

        # Spatial feature extraction
        x = self.conv(x)
        x = x.view(x.size(0), -1)       # flatten
        x = self.fc(x)                   # shared trunk

        # Dueling aggregation
        v = self.value_stream(x)          # (batch, 1)
        a = self.advantage_stream(x)      # (batch, output_size)
        return v + a - a.mean(dim=1, keepdim=True)

    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, state: np.ndarray, device: torch.device) -> int:
        """Return greedy action index for a single flat state array."""
        t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(self(t).argmax(dim=1).item())

    @torch.no_grad()
    def get_q_values(self, state: np.ndarray, device: torch.device) -> np.ndarray:
        """Return Q-values as a numpy array for a single flat state (for visualisation)."""
        t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return self(t).squeeze(0).cpu().numpy()
