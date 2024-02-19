import torch
import torch.nn as nn

from typing import Literal

class KinematicsModel(nn.Module):
    def __init__(self, n_links: int = 6, n_hidden_layers: int = 3, hidden_size: int = 32, dimensions: Literal[2, 3] = 2) -> None:
        
        super().__init__()
        self.input_layer = nn.Linear(dimensions, hidden_size)
        self.activation = nn.Tanh()
        hidden_layers = [self.activation]
        for layer in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(self.activation)
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_size, n_links * (dimensions - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input shape: (batch_size, 2) if 2D, (batch_size, 3) if 3D
        # Output shape: (batch_size, n_links) if 2D, (batch_size, n_links, 2) if 3D
        output: torch.Tensor = self.output_layer(self.hidden_layers(self.input_layer(x)))
        if x.shape[1] == 2:
            return output
        return output.view(output.shape[0], output.shape[1] // 2, 2)