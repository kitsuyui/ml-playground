import numpy as np


def test_numpy_array_basic_algebra() -> None:
    """Basic algebra on a numpy array.

    en: Example of basic algebra on a numpy array
    ja: numpy の配列に対する基本的な代数演算の見本
    """

    # en: Comparison operators (1): all elements are equal
    # ja: array_equal() を使うと配列全体の等価性を行う
    assert np.array_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))

    # en: Comparison operators (2): == is element-wise comparison
    # ja: == だけでは配列全体の等価性ではなく各成分の等価性の比較結果の配列が返る
    x = np.array([1, 2, 3]) == np.array([1, 2, 3])
    y = np.array([True, True, True])
    assert np.array_equal(x, y)

    # en: Comparison operators (3): all() is same as array_equal()
    # ja: all() も array_equal() と同様に配列全体の等価性を行う
    assert (np.array([1, 2, 3]) == np.array([1, 2, 3])).all()

    # en: Addition
    # ja: 足し算
    assert (
        np.array([1, 2, 3]) + np.array([1, 2, 3]) == np.array([2, 4, 6])
    ).all()

    # en: Subtraction
    # ja: 引き算
    assert (
        np.array([4, 5, 6]) - np.array([1, 2, 3]) == np.array([3, 3, 3])
    ).all()

    # en: Multiplication
    # ja: 各成分の掛け算
    assert (
        np.array([1, 2, 3]) * np.array([2, 3, 4]) == np.array([2, 6, 12])
    ).all()

    # en: Division
    # ja: 各成分の割り算
    assert (
        np.array([4, 5, 6]) / np.array([1, 2, 3]) == np.array([4, 2.5, 2])
    ).all()

    # en: Dot product
    # ja: 内積
    assert np.dot(np.array([1, 2, 3]), np.array([1, 2, 3])) == 14

    # en: Cross product
    # ja: 外積
    assert (
        np.cross(np.array([1, 2, 3]), np.array([4, 5, 6]))
        == np.array([-3, 6, -3])
    ).all()

    # en: Element-wise power: element-wise power
    # ja: 各成分ごとのべき乗
    assert (
        np.array([1, 2, 3]) ** np.array([2, 3, 4]) == np.array([1, 8, 81])
    ).all()

    # en: Element-wise power (2): scalar power
    # ja: スカラーのべき乗
    assert (np.array([1, 2, 3]) ** 2 == np.array([1, 4, 9])).all()
