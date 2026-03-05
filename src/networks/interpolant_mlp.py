import sys
import torch
import torch.nn as nn
from typing import List, Optional

from .mlp_base import SimpleDenseNet

class GeoPathMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        activation: str,
        batch_norm: bool = True,
        hidden_dims: Optional[List[int]] = None,
        time_geopath: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_geopath = time_geopath
        self.mainnet = SimpleDenseNet(
            input_size=2 * input_dim + (1 if time_geopath else 0),
            target_size=input_dim,
            activation=activation,
            batch_norm=batch_norm,
            hidden_dims=hidden_dims,
        )

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.time_geopath:
            x = torch.cat([x, t], dim=1)
        return self.mainnet(x)