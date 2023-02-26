import torch

from example.torch.numerical_embedding import NumericalEmbedding


def test_numerical_embedding() -> None:
    embedding_dim = 4
    ne = NumericalEmbedding(
        embedding_dim=embedding_dim,
        num_values=6,
        dropout=0.1,
    )
    x = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
    )
    assert x.shape == (6,)
    assert x.dtype == torch.float32
    y = ne(x)
    assert y.shape == (6, 4)
    assert y.dtype == torch.float32
    assert (*x.shape, embedding_dim) == y.shape

    x2 = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
        ]
    )
    assert x2.shape == (2, 6)
    assert x2.dtype == torch.float32
    y2 = ne(x2)
    assert y2.shape == (2, 6, 4)
    assert y2.dtype == torch.float32
    assert (*x2.shape, embedding_dim) == y2.shape

    # trainability
    y.sum().backward()
    assert ne.weights.grad is not None


def test_torch_jit_ready() -> None:
    """Test that the module is torch.jit.script() ready."""
    embedding_dim = 4
    ne = NumericalEmbedding(
        embedding_dim=embedding_dim,
        num_values=6,
        dropout=0.1,
    )
    ne = torch.jit.script(ne)
    x = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
    )
    assert x.shape == (6,)
    assert x.dtype == torch.float32
    y = ne(x)
    assert y.shape == (6, 4)
    assert y.dtype == torch.float32
    assert (*x.shape, embedding_dim) == y.shape
    assert ne.code is not None
