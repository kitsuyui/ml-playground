import torch

from kitsuyui_ml.torch_ext.ffn import FFN


def test_ffn() -> None:
    ffn = FFN(10, 20)
    assert ffn(torch.randn(5, 10)).shape == (5, 10)


def test_ffn_torchscript_ready() -> None:
    ffn = FFN(10, 20)
    ffn_jit = torch.jit.script(ffn)
    assert ffn_jit(torch.randn(5, 10)).shape == (5, 10)
