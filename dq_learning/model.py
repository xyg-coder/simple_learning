from __future__ import annotations

from typing import List
from typing import Sequence
from typing import cast

import torch
import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dropout > 0:
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size), nn.Dropout(p=dropout)
            )
        else:
            self.layers = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class FullyConnectedLayers(nn.Module):
    def __init__(self, output_size: int, hidden_sizes: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers = cast(
            List[nn.Module],
            [
                FullyConnectedLayer(hidden_sizes[i], hidden_sizes[i + 1], dropout=dropout)
                for i in range(len(hidden_sizes) - 1)
            ],
        ) + cast(List[nn.Module], [nn.Linear(hidden_sizes[-1], output_size)])
        self.layers = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class LinearDqn(nn.Module):
    """a deep q network consiting of fully connected layers"""

    def __init__(
        self, n_state: int, n_action: int, hidden_sizes: Sequence[int], dropout: float = 0.0, print_out_structure=False
    ) -> None:
        super().__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.fully_connected_layers = FullyConnectedLayers(self.n_action, [self.n_state] + hidden_sizes, dropout)
        if print_out_structure:
            print("model structure")
            print(self)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fully_connected_layers(tensor)

    def eval_forward_and_pick_max(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            all_actions = self.forward(tensor)
            return torch.max(all_actions, dim=1, keepdim=True).values

    def eval_forward_and_pick_max_action(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            all_actions = self.forward(tensor)
            return torch.max(all_actions, dim=1, keepdim=True).indices
