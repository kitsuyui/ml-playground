import torch
import torch.nn as nn
from torch import Tensor


class NumericalEmbedding(nn.Module):
    """
    NumericalEmbedding from FTTransformer

    Revisiting Deep Learning Models for Tabular Data https://arxiv.org/abs/2106.11959v2
    (x, num_values) -> (x, num_values, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_values: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_values, embedding_dim))
        self.biases = nn.Parameter(torch.randn(num_values, embedding_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        y = x * self.weights + self.biases
        y = self.dropout(y)
        return y
