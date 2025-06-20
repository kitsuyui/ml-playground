import torch


def test_pytorch() -> None:
    x = torch.zeros(3, 3)
    assert x.shape == (3, 3)


def test_random_with_seed() -> None:
    # without seed
    x1 = torch.rand(3, 3)
    x2 = torch.rand(3, 3)
    assert not torch.all(x1 == x2)

    # with seed
    torch.manual_seed(0)
    x1 = torch.rand(3, 3)
    torch.manual_seed(0)
    x2 = torch.rand(3, 3)
    assert torch.all(x1 == x2)
