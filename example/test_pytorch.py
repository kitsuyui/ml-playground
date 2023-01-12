import torch


def test_pytorch() -> None:
    x = torch.zeros(3, 3)
    assert x.shape == (3, 3)
