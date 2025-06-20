import torch

from kitsuyui_ml.torch_ext.scale_embedding import ScaleEmbedding, ScaleEmbedding2


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

    # Test 2
    se2 = ScaleEmbedding2(num_embeddings, embedding_dim)
    y2 = se2(x)
    assert y2.shape == y.shape


def test_torch_jit_ready() -> None:
    """Test that the module is torch.jit.script() ready."""
    num_embeddings = 100
    embedding_dim = 100
    x = torch.randint(num_embeddings, (200,))

    se = ScaleEmbedding(num_embeddings, embedding_dim)
    se = torch.jit.script(se)
    y = se(x)
    assert y.shape == (200, 100)
    assert se.code is not None

    se2 = ScaleEmbedding2(num_embeddings, embedding_dim)
    se2 = torch.jit.script(se2)
    y2 = se2(x)
    assert y2.shape == (200, 100)
    assert se2.code is not None
