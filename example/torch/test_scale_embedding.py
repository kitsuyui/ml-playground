import timeit

import torch

from example.torch.scale_embedding import ScaleEmbedding, ScaleEmbedding2


def test_scale_embedding() -> None:
    """Test scale embedding."""
    num_embeddings = 100
    embedding_dim = 100
    x = torch.randint(num_embeddings, (200,))

    # Test 1
    se = ScaleEmbedding(num_embeddings, embedding_dim)
    y = se(x)
    assert y.shape == (200, 100)

    # Test 2
    se2 = ScaleEmbedding2(num_embeddings, embedding_dim)
    y2 = se2(x)
    assert y2.shape == y.shape

    t0 = timeit.timeit("se(x)", globals=locals(), number=100000)
    t1 = timeit.timeit("se2(x)", globals=locals(), number=100000)

    # ScaleEmbedding2 uses register_buffer, but it is slower than ScaleEmbedding
    # register_buffer suits for more large constant tensors
    assert t0 < t1
