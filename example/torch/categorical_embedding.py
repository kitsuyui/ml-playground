import torch.nn as nn
from torch import Tensor

from example.torch.scale_embedding import ScaleEmbedding


class CategoricalEmbedding(nn.Module):
    """
    CategoricalEmbedding from FTTransformer

    Revisiting Deep Learning Models for Tabular Data https://arxiv.org/abs/2106.11959v2
    (x, num_categories) -> (x, num_categories, embedding_dim)
    """

    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scale_embedding = ScaleEmbedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.scale_embedding(x)
        y = self.dropout(y)
        return y  # type: ignore
