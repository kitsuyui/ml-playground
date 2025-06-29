from kitsuyui_ml.torch_ext.generate_subsequent_mask import (
    generate_square_subsequent_mask_1,
    generate_square_subsequent_mask_2,
)

import torch


def test_generate_square_subsequent_mask_1() -> None:
    mask = generate_square_subsequent_mask_1(5)
    nin = float("-inf")
    tobe = torch.Tensor(
        [
            [0.0, nin, nin, nin, nin],
            [0.0, 0.0, nin, nin, nin],
            [0.0, 0.0, 0.0, nin, nin],
            [0.0, 0.0, 0.0, 0.0, nin],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert (mask == tobe).all()
    assert mask.shape == (5, 5)
    assert mask.dtype == torch.float32
    assert mask.device == torch.device("cpu")

    # Test the edge cases when size is 0 or 1
    size_0 = 0
    mask_0 = generate_square_subsequent_mask_1(size_0)
    assert mask_0.shape == (0, 0)
    assert mask_0.dtype == torch.float32
    assert mask_0.device == torch.device("cpu")
    size_1 = 1
    mask_1 = generate_square_subsequent_mask_1(size_1)
    assert mask_1.shape == (1, 1)
    assert mask_1.dtype == torch.float32
    assert mask_1.device == torch.device("cpu")


def test_generate_square_subsequent_mask_2() -> None:
    mask = generate_square_subsequent_mask_2(5)
    nin = float("-inf")
    tobe = torch.Tensor(
        [
            [0.0, nin, nin, nin, nin],
            [0.0, 0.0, nin, nin, nin],
            [0.0, 0.0, 0.0, nin, nin],
            [0.0, 0.0, 0.0, 0.0, nin],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert (mask == tobe).all()
    assert mask.shape == (5, 5)
    assert mask.dtype == torch.float32
    assert mask.device == torch.device("cpu")
