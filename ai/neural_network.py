"""Réseau de neurones pour le DQN — architecture Dueling."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuralNetwork(nn.Module):
    """
    Dueling Deep Q-Network (Wang et al. 2016).

    Shared feature extractor feeds into two independent streams:
      • Value stream     V(s)      — how good is this state?
      • Advantage stream A(s, a)   — how much better is action a vs average?

    Combined:  Q(s, a) = V(s) + A(s, a) − mean_a'[ A(s, a') ]

    Subtracting the mean advantage makes V and A identifiable and
    stabilises training, especially when many actions are equivalent.
    """

    def __init__(
        self,
        input_size:  int = 14,
        hidden_size: int = 256,
        output_size: int = 4,
    ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        stream_size = hidden_size // 2   # 128

        # Shared feature extractor
        self.fc1 = nn.Linear(input_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Value stream: hidden → 128 → 1
        self.value_fc  = nn.Linear(hidden_size, stream_size)
        self.value_out = nn.Linear(stream_size, 1)

        # Advantage stream: hidden → 128 → output_size
        self.adv_fc  = nn.Linear(hidden_size, stream_size)
        self.adv_out = nn.Linear(stream_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.fc1, self.fc2,
                      self.value_fc, self.value_out,
                      self.adv_fc,   self.adv_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_size)
        Returns:
            Q-values: (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)                        # (batch, 1)

        # Advantage stream
        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)                          # (batch, output_size)

        # Dueling aggregation — subtract mean advantage for identifiability
        return v + a - a.mean(dim=1, keepdim=True)

    def predict(self, state: np.ndarray, device: torch.device) -> int:
        """Return the greedy action index for a single state."""
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.forward(t).argmax(dim=1).item()

    def get_q_values(self, state: np.ndarray, device: torch.device) -> np.ndarray:
        """Return all Q-values for a single state (for visualisation)."""
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.forward(t).cpu().numpy().flatten()
