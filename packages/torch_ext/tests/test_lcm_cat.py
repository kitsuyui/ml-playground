import torch

from kitsuyui_ml.torch_ext.lcm_cat import lcm_cat


def test_lcm_cat_batch_first_false() -> None:
    asis = lcm_cat(
        [
            torch.full((2, 3, 2, 3), 8),
            torch.full((1, 3, 3, 1), 9),
        ]
    )
    tobe = torch.tensor(
        [
            [
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
            ],
            [
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
            ],
        ]
    )
    assert torch.all(asis == tobe)
    assert asis.shape == tobe.shape == (2, 3, 9)


def test_lcm_cat_batch_first_true() -> None:
    asis = lcm_cat(
        [
            torch.full((3, 2, 2, 3), 8),
            torch.full((3, 1, 3, 1), 9),
        ],
        batch_first=True,
    )
    tobe = torch.tensor(
        [
            [
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
            ],
            [
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
            ],
            [
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
                [8, 8, 8, 8, 8, 8, 9, 9, 9],
            ],
        ]
    )
    assert torch.all(asis == tobe)
    assert asis.shape == tobe.shape == (3, 2, 9)


def test_lcm_cat_torch_jit_ready() -> None:
    lcm_cat_jit = torch.jit.script(lcm_cat)
    asis = lcm_cat_jit(
        [
            torch.full((2, 3, 2, 3), 8),
            torch.full((1, 3, 3, 1), 9),
        ]
    )
    assert asis.shape == (2, 3, 9)

    asis = lcm_cat_jit(
        [
            torch.full((3, 2, 2, 3), 8),
            torch.full((3, 1, 3, 1), 9),
        ],
        batch_first=True,
    )
    assert asis.shape == (3, 2, 9)
