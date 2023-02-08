import torch

from example.torch.positional_encoding import PositionalEncoding


def test_positional_encoding() -> None:
    pe = PositionalEncoding(d_model=4, dropout=0.0)
    x = torch.Tensor(
        [1.0, 2.0, 3.0, 4.0],
    )
    y = pe(x)
    # The difference should be between -1.0 and 1.0
    assert torch.allclose(x, y, rtol=0.0, atol=1.0)
