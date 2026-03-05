import sys
import torch
import torch.nn as nn
from typing import List, Optional

from .mlp_base import SimpleDenseNet

class GrowthNet(SimpleDenseNet):
    def __init__(
        self,
        dim: int,
        activation: str,
        hidden_dims: List[int] = None,
        batch_norm: bool = False,
        negative: bool = False
    ):
        super().__init__(input_size=dim + 1, target_size=1,
                         activation=activation, 
                         batch_norm=batch_norm, 
                         hidden_dims=hidden_dims)
        
        self.softplus = nn.Softplus()
        self.negative = negative
        
    def forward(self, t, x):

        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        x = torch.cat([t, x], dim=-1)
        x = self.model(x)
        x = self.softplus(x)
        if self.negative:
            x = -x
        return x