import torch

from example.torch.ffn import FFN


def test_ffn() -> None:
    ffn = FFN(10, 20)
    assert ffn(torch.randn(5, 10)).shape == (5, 10)


def test_ffn_torchscript_ready() -> None:
    ffn = FFN(10, 20)
    ffn = torch.jit.script(ffn)
    assert ffn(torch.randn(5, 10)).shape == (5, 10)
