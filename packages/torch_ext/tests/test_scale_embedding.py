import subprocess
import sys

import pytest
import torch

from kitsuyui_ml.torch_ext.scale_embedding import (
    ScaleEmbedding,
    ScaleEmbedding2,
)


def test_scale_embedding() -> None:
    """Test scale embedding."""
    num_embeddings = 10
    embedding_dim = 20
    input_dim = 2
    x = torch.randint(num_embeddings, (input_dim,))

    # Test 1
    se = ScaleEmbedding(num_embeddings, embedding_dim)
    y = se(x)
    assert y.shape == (input_dim, embedding_dim)
    assert "scale" in se.state_dict()

    se = se.to(dtype=torch.float64)
    assert se.scale.dtype == torch.float64

    # Test 2
    with pytest.warns(FutureWarning, match="ScaleEmbedding2 is deprecated"):
        se2 = ScaleEmbedding2(num_embeddings, embedding_dim)
    y2 = se2(x)
    assert y2.shape == y.shape


def test_torch_jit_ready() -> None:
    """Test that the module is torch.jit.script() ready."""
    num_embeddings = 100
    embedding_dim = 100
    x = torch.randint(num_embeddings, (200,))

    se = ScaleEmbedding(num_embeddings, embedding_dim)
    se_jit = torch.jit.script(se)
    y = se_jit(x)
    assert y.shape == (200, 100)
    assert se_jit.code is not None

    with pytest.warns(FutureWarning, match="ScaleEmbedding2 is deprecated"):
        se2 = ScaleEmbedding2(num_embeddings, embedding_dim)
    se2_jit = torch.jit.script(se2)
    y2 = se2_jit(x)
    assert y2.shape == (200, 100)
    assert se2_jit.code is not None


def test_scale_embedding2_warning_is_visible_from_imported_module(
    tmp_path: pytest.TempPathFactory,
) -> None:
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from kitsuyui_ml.torch_ext.scale_embedding import ScaleEmbedding2\n"
        "ScaleEmbedding2(num_embeddings=10, embedding_dim=4)\n",
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
    assert "FutureWarning: ScaleEmbedding2 is deprecated" in proc.stderr
