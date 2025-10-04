import numpy as np
import torch
from torch import nn


class TorchBrain(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weights: np.ndarray | None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if weights is not None:
            self.set_weights(weights)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x * (np.pi / 2)
        # safety clamp for out of range values
        x = torch.clip(x, min=-np.pi / 2, max=np.pi / 2)
        return x.numpy()

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """Load weights from a 1D tensor or numpy array into the model."""
        flat_weights = torch.tensor(flat_weights, dtype=torch.float32)
        idx = 0
        for p in self.parameters():
            num_params = p.numel()
            p.data = flat_weights[idx : idx + num_params].view_as(p).clone()
            idx += num_params
