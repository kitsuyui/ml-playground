import timeit

import torch
import torch.nn as nn

# https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html


def something_orig(x: torch.Tensor) -> torch.Tensor:
    for i in range(512):
        x += i
    return x


@torch.jit.script
def something_jit(x: torch.Tensor) -> torch.Tensor:
    for i in range(512):
        x += i
    return x


class SomethingOrig(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(512):
            x += i
        return x


def test_torch_jit_fn() -> None:
    x = torch.rand(1)
    assert something_orig(x) == something_jit(x)
    orig_time = timeit.timeit(lambda: something_orig(x), number=1000)
    jit_time = timeit.timeit(lambda: something_jit(x), number=1000)

    tobe_ir = """\
def something_jit(x: Tensor) -> Tensor:
  x0 = x
  for i in range(512):
    x0 = torch.add_(x0, i)
  return x0
"""
    assert something_jit.code == tobe_ir
    assert (
        jit_time < orig_time
    ), f"JIT is must faster than original: but (orig, jit) = ({orig_time}, {jit_time})"


def test_torch_jit_nn_module() -> None:
    x = torch.rand(1)
    orig = SomethingOrig()
    jit = torch.jit.script(orig)
    assert orig(x) == jit(x)

    orig_time = timeit.timeit(lambda: orig(x), number=1000)
    jit_time = timeit.timeit(lambda: jit(x), number=1000)

    tobe_ir = """\
def forward(self,
    x: Tensor) -> Tensor:
  x0 = x
  for i in range(512):
    x0 = torch.add_(x0, i)
  return x0
"""
    print(tobe_ir)
    assert jit.code == tobe_ir

    assert (
        jit_time < orig_time
    ), f"JIT is must faster than original: but (orig, jit) = ({orig_time}, {jit_time})"
