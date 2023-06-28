import torch
import torch.nn as nn

# https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html


def something(x: torch.Tensor) -> torch.Tensor:
    for i in range(512):
        x += i
    return x


something_jit = torch.jit.script(something)


class SomethingOrig(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(512):
            x += i
        return x


def test_torch_jit_fn() -> None:
    x = torch.rand(1)
    assert something(x) == something_jit(x)
    tobe_ir = """\
def something(x: Tensor) -> Tensor:
  x0 = x
  for i in range(512):
    x0 = torch.add_(x0, i)
  return x0
"""
    assert something_jit.code == tobe_ir


def test_torch_jit_nn_module() -> None:
    x = torch.rand(1)
    orig = SomethingOrig()
    jit = torch.jit.script(orig)
    assert orig(x) == jit(x)

    tobe_ir = """\
def forward(self,
    x: Tensor) -> Tensor:
  x0 = x
  for i in range(512):
    x0 = torch.add_(x0, i)
  return x0
"""
    assert jit.code == tobe_ir
