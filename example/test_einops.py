import einops
import numpy as np


def test_einops_rearrange_numpy() -> None:
    """einops.rearrange() for numpy.ndarray

    en: rearrange axes of an array
    ja: 配列の軸を入れ替える
    """
    # en: reshape (2x3x4) array to (2x4x3) array
    # ja: (2x3x4) の配列を (2x4x3) に変換する
    x = np.array(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
            ],
        ]
    )
    assert x.shape == (2, 3, 4)
    y = einops.rearrange(x, "a b c -> a c b", a=2, b=3, c=4)
    y_tobe = np.array(
        [
            [
                [1, 5, 9],
                [2, 6, 10],
                [3, 7, 11],
                [4, 8, 12],
            ],
            [
                [13, 17, 21],
                [14, 18, 22],
                [15, 19, 23],
                [16, 20, 24],
            ],
        ]
    )
    assert y.shape == (2, 4, 3)
    assert (y == y_tobe).all()

    # en: reshape (2 x 3 x 4) array to (8 x 3) array
    # ja: (2 x 3 x 4) の配列を (8 x 3) に変換する
    y = einops.rearrange(x, "a b c -> (a c) b", a=2, b=3, c=4)
    y_tobe = np.array(
        [
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11],
            [4, 8, 12],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23],
            [16, 20, 24],
        ]
    )
    assert y.shape == (8, 3)
    assert (y == y_tobe).all()

    # en: reshape (8x3) array to (2x4x3) array
    # ja: (8x3) の配列を (2x4x3) に変換する
    x = np.array(
        [
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11],
            [4, 8, 12],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23],
            [16, 20, 24],
        ]
    )
    y = einops.rearrange(x, "(a c) b -> a c b", a=2, b=3, c=4)
    y_tobe = np.array(
        [
            [
                [1, 5, 9],
                [2, 6, 10],
                [3, 7, 11],
                [4, 8, 12],
            ],
            [
                [13, 17, 21],
                [14, 18, 22],
                [15, 19, 23],
                [16, 20, 24],
            ],
        ]
    )
