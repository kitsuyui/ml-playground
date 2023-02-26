import torch

from example.torch.token_embedding import TokenEmbedding


def test_token_embedding() -> None:
    embedding_dim = 4
    te = TokenEmbedding(
        num_vocab=6,
        embedding_dim=embedding_dim,
        dropout=0.1,
    )

    x = torch.tensor(
        [1, 2, 3],
    )
    assert x.shape == (3,)
    assert x.dtype == torch.int64
    y = te(x)
    assert y.shape == (3, 4)
    assert y.dtype == torch.float32
    assert (*x.shape, embedding_dim) == y.shape

    x2 = torch.tensor(
        [
            [1, 2, 3, 4, 5, 2, 3],
            [1, 2, 3, 4, 5, 2, 3],
        ]
    )
    assert x2.shape == (2, 7)
    assert x2.dtype == torch.int64
    y2 = te(x2)
    assert y2.shape == (2, 7, 4)
    assert y2.dtype == torch.float32
    assert (*x2.shape, embedding_dim) == y2.shape

    # trainability
    y.sum().backward()
    assert te.scale_embedding.embedding.weight.grad is not None


def test_torch_jit_ready() -> None:
    """Test that the module is torch.jit.script() ready."""
    embedding_dim = 4
    te = TokenEmbedding(
        num_vocab=6,
        embedding_dim=embedding_dim,
        dropout=0.1,
    )
    te = torch.jit.script(te)
    x = torch.tensor(
        [1, 2, 3],
    )
    assert x.shape == (3,)
    assert x.dtype == torch.int64
    y = te(x)
    assert y.shape == (3, 4)
    assert y.dtype == torch.float32
    assert (*x.shape, embedding_dim) == y.shape
    assert te.code is not None
