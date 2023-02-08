import torch
import torch.nn as nn
from torch import Tensor


class ScaleEmbedding(nn.Module):
    """ScaleEmbedding"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.scale = torch.sqrt(
            torch.tensor(embedding_dim, requires_grad=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x.long()) * self.scale  # type: ignore


class ScaleEmbedding2(nn.Module):
    """ScaleEmbedding"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.register_buffer("scale", torch.sqrt(torch.tensor(embedding_dim)))

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x.long()) * self.scale  # type: ignore
