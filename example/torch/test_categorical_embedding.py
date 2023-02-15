import torch

from example.torch.categorical_embedding import CategoricalEmbedding


def test_categorical_embedding() -> None:
    embedding_dim = 4
    ce = CategoricalEmbedding(
        num_categories=6,
        embedding_dim=embedding_dim,
        dropout=0.1,
    )

    x = torch.tensor(
        [1, 2, 3],
    )
    assert x.shape == (3,)
    assert x.dtype == torch.int64
    y = ce(x)
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
    y2 = ce(x2)
    assert y2.shape == (2, 7, 4)
    assert y2.dtype == torch.float32
    assert (*x2.shape, embedding_dim) == y2.shape

    # trainability
    y.sum().backward()
    assert ce.scale_embedding.embedding.weight.grad is not None
