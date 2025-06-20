# Position-wise Feed-Forward Networks

import torch
import torch.nn.functional as F
from torch import nn


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = self.linear1(x)
        ys = F.relu(ys)
        return self.linear2(ys)  # type: ignore


FFN = PositionWiseFeedForwardNetwork
