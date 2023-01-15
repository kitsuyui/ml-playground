import random


def test_random_with_seed() -> None:
    """Test that random.random() is deterministic with seed."""

    # without seed
    x1 = random.random()
    x2 = random.random()
    assert x1 != x2

    # with seed
    random.seed(0)
    x1 = random.random()
    random.seed(0)
    x2 = random.random()
    assert x1 == x2
