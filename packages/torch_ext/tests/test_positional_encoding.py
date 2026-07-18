import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kitsuyui_ml.torch_ext.positional_encoding import (
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
    with pytest.warns(FutureWarning, match="PositionalEncoding2 is deprecated"):
        pe2 = PositionalEncoding2(d_model=4, dropout=0.0)
    assert (
        repr(pe2)
        == """\
PositionalEncoding2(
  (seq): Sequential(
    (pe): RawPositionalEncoding()
    (dropout): Dropout(p=0.0, inplace=False)
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


def test_positional_encoding2_drops_positional_signal() -> None:
    x = torch.ones(3, 2, 4)
    pe = PositionalEncoding(d_model=4, dropout=1.0)
    with pytest.warns(FutureWarning, match="PositionalEncoding2 is deprecated"):
        pe2 = PositionalEncoding2(d_model=4, dropout=1.0)

    expected = torch.zeros_like(x)
    assert torch.equal(pe(x), expected)
    assert torch.equal(pe2(x), expected)


def test_torch_jit_ready() -> None:
    """Test that the module is torch.jit.script() ready."""
    pe = PositionalEncoding(d_model=4, dropout=0.0)
    pe_jit = torch.jit.script(pe)  # type: ignore
    x = torch.Tensor(
        [1.0, 2.0, 3.0, 4.0],
    )
    y = pe_jit(x)
    assert y.shape == (4, 1, 4)
    assert pe_jit.code is not None

    with pytest.warns(FutureWarning, match="PositionalEncoding2 is deprecated"):
        pe2 = PositionalEncoding2(d_model=4, dropout=0.0)
    pe2_jit = torch.jit.script(pe2)
    y2 = pe2_jit(x)
    assert y2.shape == (4, 1, 4)
    assert pe2_jit.code is not None


def test_positional_encoding2_warning_is_visible_from_imported_module(
    tmp_path: Path,
) -> None:
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from kitsuyui_ml.torch_ext.positional_encoding import PositionalEncoding2\n"
        "PositionalEncoding2(d_model=4, dropout=0.0)\n",
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import consumer\n",
        encoding="utf-8",
    )

    proc = subprocess.run(  # noqa: S603 - fixed interpreter and temp script path
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert proc.returncode == 0
    assert "FutureWarning: PositionalEncoding2 is deprecated" in proc.stderr
