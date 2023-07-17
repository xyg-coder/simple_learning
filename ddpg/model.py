from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(400, action_dim)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        hidden_output = self.hidden_layers(state_tensor)
        output = self.output_layer(hidden_output)
        return nn.functional.tanh(output)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(400, 1)
        nn.init.uniform_(self.output_layer.weight, -3e-4, 3e-4)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        hidden_output = self.hidden_layers(torch.cat([state_tensor, action_tensor], dim=1))
        return self.output_layer(hidden_output)
