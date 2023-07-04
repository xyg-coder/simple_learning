from __future__ import annotations

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    """
    input: [n_batch, n_channel, 80, 80]
    output: [n_batch, 1]
    """

    def __init__(self, n_channel: int) -> None:
        super().__init__()
        # output: (80-3)/2+1=39
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channel, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(4),
        )
        # output: (39-3)/2+1=19
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # output: (19-3)/2+1=9
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear_size = 9 * 9 * 16
        self.linear1 = nn.Sequential(nn.Linear(self.linear_size, 256), nn.ReLU())
        self.linear2 = nn.Linear(256, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        n_batch = batch.shape[0]
        output = self.conv1(batch)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(n_batch, -1)
        output = self.linear1(output)
        output = self.linear2(output)
        return torch.sigmoid(output)
