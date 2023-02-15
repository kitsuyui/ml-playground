import torch.nn as nn
from torch import Tensor

from example.torch.positional_encoding import PositionalEncoding
from example.torch.scale_embedding import ScaleEmbedding


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scale_embedding = ScaleEmbedding(
            num_embeddings=num_vocab,
            embedding_dim=embedding_dim,
        )
        self.pe = PositionalEncoding(
            d_model=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.scale_embedding(x)
        y = self.pe(y)
        return y  # type: ignore
