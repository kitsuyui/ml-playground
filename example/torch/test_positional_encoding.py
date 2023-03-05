import timeit

import torch

from example.torch.positional_encoding import (
    PositionalEncoding,
    PositionalEncoding2,
)


def test_positional_encoding() -> None:
    pe = PositionalEncoding(d_model=4, dropout=0.0)
    x = torch.Tensor(
        [1.0, 2.0, 3.0, 4.0],
    )
    y = pe(x)
    # The difference should be between -1.0 and 1.0
    assert torch.allclose(x, y, rtol=0.0, atol=1.0)


def test_positional_encoding_repr() -> None:
    pe = PositionalEncoding(d_model=4, dropout=0.0)
    assert (
        repr(pe)
        == """\
PositionalEncoding(
  (dropout): Dropout(p=0.0, inplace=False)
)"""
    )
    pe2 = PositionalEncoding2(d_model=4, dropout=0.0)
    assert (
        repr(pe2)
        == """\
PositionalEncoding2(
  (seq): Sequential(
    (dropout): Dropout(p=0.0, inplace=False)
    (pe): RawPositionalEncoding()
  )
)"""
    )
    x = torch.Tensor(
        [1.0, 2.0, 3.0, 4.0],
    )
    y = pe(x)
    y2 = pe2(x)
    # same result
    assert torch.allclose(y, y2, rtol=0.0, atol=0.0)

    # benchmark
    t0 = timeit.timeit("pe(x)", globals=locals(), number=200000)
    t1 = timeit.timeit("pe2(x)", globals=locals(), number=200000)

    # the original implementation is faster because it is simpler (it does not use Sequential module)
    assert t0 < t1


def test_torch_jit_ready() -> None:
    """Test that the module is torch.jit.script() ready."""
    pe = PositionalEncoding(d_model=4, dropout=0.0)
    pe = torch.jit.script(pe)
    x = torch.Tensor(
        [1.0, 2.0, 3.0, 4.0],
    )
    y = pe(x)
    assert y.shape == (4, 1, 4)
    assert pe.code is not None

    pe2 = PositionalEncoding2(d_model=4, dropout=0.0)
    pe2 = torch.jit.script(pe2)
    y2 = pe2(x)
    assert y2.shape == (4, 1, 4)
    assert pe2.code is not None
