import math
import warnings
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
    """PositionalEncoding without the dropout layer.

    Internal building block used by PositionalEncoding2.
    """

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
    """Deprecated. Use PositionalEncoding instead.

    PositionalEncoding2 is equivalent to PositionalEncoding and will be removed in a future version.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        warnings.warn(
            "PositionalEncoding2 is deprecated and equivalent to PositionalEncoding. "
            "Use PositionalEncoding instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__()
        self.seq = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pe",
                        RawPositionalEncoding(d_model, max_len),
                    ),
                    ("dropout", nn.Dropout(p=dropout)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.seq(x)  # type: ignore
