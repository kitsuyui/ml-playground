import math
from typing import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Original PositionalEncoding from the pytorch tutorial.

    from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if len(x.shape) == 2:
            x = x + self.pe[: x.size(0)].squeeze(1)  # type: ignore
            return self.dropout(x)  # type: ignore

        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)  # type: ignore


class RawPositionalEncoding(nn.Module):
    """PositionalEncoding without the dropout layer."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]  # type: ignore
        return x  # type: ignore


class PositionalEncoding2(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            OrderedDict(
                [
                    ("dropout", nn.Dropout(p=dropout)),
                    (
                        "pe",
                        RawPositionalEncoding(d_model, max_len),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.seq(x)  # type: ignore
