import torch
from torch.nn import Transformer


def test_generate_square_subsequent_mask_mps() -> None:
    # Test generate_square_subsequent_mask works fine on cpu
    tobe = torch.tensor([[0.0, -float("inf")], [0.0, 0.0]])
    assert (
        Transformer.generate_square_subsequent_mask(2, device="cpu") == tobe
    ).all()
    assert (Transformer.generate_square_subsequent_mask(2) == tobe).all()

    # But on mps, it will return nan
    if torch.backends.mps.is_available():
        actual = torch.tensor(
            [[float("nan"), -float("inf")], [float("nan"), float("nan")]],
            device="mps",
        )
        result = Transformer.generate_square_subsequent_mask(2, device="mps")
        assert torch.allclose(result, actual, equal_nan=True)

    # Conclusion: generate_square_subsequent_mask is not supported on mps
    # When we use generate_square_subsequent_mask, we should use cpu and convert to mps with to(device="mps")
