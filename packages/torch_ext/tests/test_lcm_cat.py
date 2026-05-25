import pytest
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


def test_lcm_cat_rejects_default_sequence_expansion_over_cap() -> None:
    with pytest.raises(
        ValueError,
        match="lcm_cat sequence length exceeds max_sequence_length",
    ):
        lcm_cat(
            [
                torch.full((4097, 1, 1), 8),
                torch.full((4099, 1, 1), 9),
            ]
        )


def test_lcm_cat_allows_custom_sequence_cap() -> None:
    asis = lcm_cat(
        [
            torch.full((4, 1, 1), 8),
            torch.full((5, 1, 1), 9),
        ],
        max_sequence_length=20,
    )
    assert asis.shape == (20, 1, 2)


def test_lcm_cat_rejects_non_positive_sequence_cap() -> None:
    with pytest.raises(
        ValueError,
        match="max_sequence_length must be positive",
    ):
        lcm_cat([torch.full((2, 1, 1), 8)], max_sequence_length=0)


def test_lcm_cat_torch_jit_rejects_sequence_expansion_over_cap() -> None:
    lcm_cat_jit = torch.jit.script(lcm_cat)
    with pytest.raises(torch.jit.Error, match="max_sequence_length"):
        lcm_cat_jit(
            [
                torch.full((17, 1, 1), 8),
                torch.full((19, 1, 1), 9),
            ],
            max_sequence_length=100,
        )
