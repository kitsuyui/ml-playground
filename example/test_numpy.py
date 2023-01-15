import numpy as np
import pytest


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


def test_numpy_types() -> None:
    """Types of numpy arrays.

    en: Example of types of numpy arrays
    ja: numpy の配列の型の見本
    """

    # int64
    x = np.array(
        [1, 2, 3, 4, 5]
    )  # en: Python's int list is converted to int64 by default
    assert x.dtype == np.int64
    assert x.size == 5, "array size is 5"
    assert x.itemsize == 8, "int64 is 8 bytes"
    assert x.size * x.itemsize == 40, "array byte size is 40"

    # float64
    x = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )  # en: Python's float list is converted to float64 by default
    assert x.dtype == np.float64
    assert x.size == 5, "size is 5"
    assert x.itemsize == 8, "float64 is 8 bytes"
    assert x.size * x.itemsize == 40, "array byte size is 40"

    # complex128
    x = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j, 5.0 + 5.0j])
    assert x.dtype == np.complex128
    assert x.size == 5, "size is 5"
    assert x.itemsize == 16, "complex128 is 16 bytes"
    assert x.size * x.itemsize == 80, "array byte size is 80"

    # int32
    x = np.array(
        [1, 2, 3, 4, 5], dtype=np.int32
    )  # explicit type specification is required
    assert x.dtype == np.int32
    assert x.size == 5, "size is 5"
    assert x.itemsize == 4, "int32 is 4 bytes"
    assert x.size * x.itemsize == 20, "array byte size is 20"

    # float32
    x = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
    )  # explicit type specification is required
    assert x.dtype == np.float32
    assert x.size == 5, "size is 5"
    assert x.itemsize == 4, "float32 is 4 bytes"
    assert x.size * x.itemsize == 20, "array byte size is 20"

    # complex64
    x = np.array(
        [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j, 5.0 + 5.0j],
        dtype=np.complex64,
    )  # explicit type specification is required
    assert x.dtype == np.complex64
    assert x.size == 5, "size is 5"
    assert x.itemsize == 8, "complex64 is 8 bytes"
    assert x.size * x.itemsize == 40, "array byte size is 40"

    # mixed
    x = np.array(
        [1, 2, 3, 4, 5.0]
    )  # en: mixed int and float is converted to float
    assert x.dtype == np.float64
    assert x.size == 5, "size is 5"
    assert x.itemsize == 8, "float64 is 8 bytes"
    assert x.size * x.itemsize == 40, "array byte size is 40"

    # string (fixed length)
    x = np.array(["a", "b", "c", "d", "e"])
    assert x.dtype == np.dtype("U1")
    assert x.size == 5, "size is 5"
    assert x.itemsize == 4, "U1 is 4 byte"
    assert x.size * x.itemsize == 20, "array byte size is 20"

    # string (variable length)
    x = np.array(
        ["a", "bb", "ccc", "dddd", "eeeee"]
    )  # en: The length of the string with the maximum length is used
    assert x.dtype == np.dtype("U5")
    assert x.size == 5, "size is 5"
    assert x.itemsize == 20, "U5 is 20 bytes"
    assert x.size * x.itemsize == 100, "array byte size is 100"

    # unicode (fixed length)
    x = np.array(["あ", "い", "う", "え", "お"])
    assert x.dtype == np.dtype("U1")
    assert x.size == 5, "size is 5"
    assert x.itemsize == 4, "U1 is 1 byte"
    assert x.size * x.itemsize == 20, "array byte size is 20"

    # unicode (Emoji ZWJ Sequence)
    # Emoji ZWJ Sequence is a sequence of Unicode characters and
    # ZERO WIDTH JOINER (U+200D) that is used to represent a single emoji.
    # Therefore, the length of a single character is not 1 but 7
    x = np.array(["👨‍👩‍👧‍👦", "👨‍👩‍👧‍👦", "👨‍👩‍👧‍👦", "👨‍👩‍👧‍👦", "👨‍👩‍👧‍👦"])
    assert x.dtype == np.dtype("<U7")
    assert x.size == 5, "size is 5"
    assert x.itemsize == 28, "U7 is 28 bytes"
    assert x.size * x.itemsize == 140, "array byte size is 140"

    # addition of float and int
    # ja: int と float で演算を行う場合に左右の計算順序には関係なく float64 に変換する
    # en: When performing operations on int and float, regardless of order,
    # en: it is converted to float64
    x = np.array([1, 2, 3, 4, 5]) + np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert x.dtype == np.float64
    assert x.size == 5, "size is 5"
    assert x.itemsize == 8, "float64 is 8 bytes"
    assert x.size * x.itemsize == 40, "array byte size is 40"
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + np.array([1, 2, 3, 4, 5])
    assert x.dtype == np.float64
    assert x.size == 5, "size is 5"
    assert x.itemsize == 8, "float64 is 8 bytes"
    assert x.size * x.itemsize == 40, "array byte size is 40"


def test_numpy_n_dimention_array() -> None:
    """numpy.ndarray

    en: numpy.ndarray as n-dimensional array/tensor
    ja: numpy.ndarray を n 次元配列・テンソルとして扱う
    """

    # 1D tensor (vector)
    x = np.array([1, 2, 3])
    assert x.ndim == 1  # number of dimensions
    assert x.shape == (3,)  # shape of array
    assert x.size == 3  # number of elements
    assert x.dtype == np.int64  # data type
    assert x.itemsize == 8  # size of each element
    assert x.nbytes == 24  # total size of array

    # 2D tensor (matrix)
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.ndim == 2
    assert x.shape == (2, 3)  # en: (row, column), ja: (行, 列)
    assert x.size == 6
    assert x.dtype == np.int64
    assert x.itemsize == 8
    assert x.nbytes == 48

    # 3D tensor
    x = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ]
    )
    assert x.ndim == 3
    assert x.shape == (2, 3, 4)
    assert x.size == 24
    assert x.dtype == np.int64
    assert x.itemsize == 8
    assert x.nbytes == 192


def test_numpy_matrix_operations() -> None:
    """Matrix operations with numpy.ndarray

    en: Matrix operations with numpy.ndarray
    ja: numpy.ndarray による行列演算
    """

    # transpose
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.shape == (2, 3)
    x_t = x.T  # en: transpose, ja: 転置
    assert x.T.shape == (3, 2)
    x_t_tobe = np.array(
        [
            [1, 4],
            [2, 5],
            [3, 6],
        ]
    )
    assert (x_t == x_t_tobe).all()

    # dot product
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    y = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )
    assert x.shape == (2, 3)
    assert y.shape == (3, 2)
    z = np.dot(x, y)  # en: dot product, ja: 内積
    assert z.shape == (2, 2)
    z_tobe = np.array(
        [
            [22, 28],
            [49, 64],
        ]
    )
    assert (z == z_tobe).all()

    # dot product (shorter way)
    z2 = x @ y  # en: dot product, ja: 内積
    assert (z2 == z_tobe).all()

    # element-wise multiplication
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    y = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    z = x * y  # en: element-wise multiplication, ja: 要素ごとの積
    assert z.shape == (2, 3)

    # reshape
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.shape == (2, 3)
    x_reshaped = x.reshape(3, 2)  # en: reshape, ja: 変形
    assert x_reshaped.shape == (3, 2)
    x_reshaped_tobe = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )
    assert (x_reshaped == x_reshaped_tobe).all()


def test_numpy_convenient_methods() -> None:
    # zeros
    # en: create an array filled with zeros
    # ja: 0 で埋められた配列を作成する
    x = np.zeros((2, 3))
    assert x.shape == (2, 3)
    assert (x == np.array([[0, 0, 0], [0, 0, 0]])).all()

    # ones
    # en: create an array filled with ones
    # ja: 1 で埋められた配列を作成する
    x = np.ones((2, 3))
    assert x.shape == (2, 3)
    assert (x == np.array([[1, 1, 1], [1, 1, 1]])).all()

    # full
    # en: create an array filled with a given value
    # ja: 指定した値で埋められた配列を作成する
    x = np.full((2, 3), 3.14)
    assert x.shape == (2, 3)
    assert (x == np.array([[3.14, 3.14, 3.14], [3.14, 3.14, 3.14]])).all()

    # random
    # en: create an array filled with random values
    # ja: ランダムな値で埋められた配列を作成する
    x = np.random.random((2, 3))
    assert x.shape == (2, 3)
    assert x.dtype == np.float64

    # sum
    # en: sum of all elements
    # ja: 全要素の和
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.sum() == 21

    # mean
    # en: mean of all elements
    # ja: 全要素の平均
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.mean() == 3.5

    # std
    # en: standard deviation of all elements
    # ja: 全要素の標準偏差
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.std() == 1.707825127659933

    # max
    # en: maximum value of all elements
    # ja: 全要素の最大値
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.max() == 6

    # min
    # en: minimum value of all elements
    # ja: 全要素の最小値
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    assert x.min() == 1

    # like functions
    # en: functions that return an array with the same shape as the input array
    # ja: 入力配列と同じ形状の配列を返す関数

    # zeros_like
    # en: create an array filled with zeros
    # ja: 0 で埋められた配列を作成する
    x = np.array([1, 2, 3])
    x_zeros_like = np.zeros_like(x)
    assert x_zeros_like.shape == (3,)
    assert (x_zeros_like == np.array([0, 0, 0])).all()

    # ones_like
    # en: create an array filled with ones
    # ja: 1 で埋められた配列を作成する
    x = np.array([1, 2, 3])
    x_ones_like = np.ones_like(x)
    assert x_ones_like.shape == (3,)
    assert (x_ones_like == np.array([1, 1, 1])).all()

    # full_like
    # en: create an array filled with a given value
    # ja: 指定した値で埋められた配列を作成する
    x = np.array([1, 2, 3])
    x_full_like = np.full_like(x, 3.14, dtype=np.float64)
    assert x_full_like.shape == (3,)
    assert (x_full_like == np.array([3.14, 3.14, 3.14])).all()


def test_numpy_indexing_and_slices() -> None:
    """Indexing and slices

    ja: numpy の多次元配列のインデクスにはおもしろい記法があるので注意する
    en: numpy has some interesting indexing syntax for multidimensional arrays
    """

    # n-th element
    # en: get the n-th element
    # ja: n 番目の要素を取得する
    x = np.array([111, 222, 333])
    assert x[0] == 111
    assert x[1] == 222
    assert x[2] == 333
    assert x[0].dtype == np.int64

    # n-th element (negative index)
    # en: get the n-th element from the end
    # ja: 負のインデックスは末尾からのインデックス
    x = np.array([111, 222, 333])
    assert x[-1] == 333
    assert x[-2] == 222
    assert x[-3] == 111

    # n-th element (over index)
    # en: get the n-th element from the end
    # ja: インデックスが配列の長さを超えるとエラーになる
    x = np.array([111, 222, 333])
    with pytest.raises(IndexError):
        x[3]

    # n-th element (over negative index)
    # en: get the n-th element from the end
    # ja: 負のインデックスが配列の長さを超えるとエラーになる
    x = np.array([111, 222, 333])
    with pytest.raises(IndexError):
        x[-4]

    # slice
    # en: get a slice of the array
    # ja: 配列のスライスを取得する
    x = np.array([111, 222, 333])
    assert (x[0:2] == np.array([111, 222])).all()
    assert (x[1:3] == np.array([222, 333])).all()
    assert (x[0:3] == np.array([111, 222, 333])).all()
    assert (x[0:4] == np.array([111, 222, 333])).all()
    assert (x[0:0] == np.array([])).all()
    assert (x[0:-1] == np.array([111, 222])).all()
    assert (x[0:-2] == np.array([111])).all()
    assert (x[0:-3] == np.array([])).all()

    # n-dim array
    # en: get the n-th element of the n-dim array
    # ja: n 次元配列の n 番目の要素を取得する
    x = np.array(
        [
            [111, 222, 333],
            [444, 555, 666],
        ]
    )
    # en: get the 1st element of the 1st row. Python's list is x[0][0]
    # en: but this is more readable
    # ja: n 行 n 列の要素を取得する。Python のリストの x[0][0] と同じだが,
    # ja: 慣れればこちらの方がわかりやすい
    assert x[0, 0] == 111
    assert x[0, 1] == 222
    assert x[0, 2] == 333
    assert x[1, 0] == 444
    assert x[1, 1] == 555
    assert x[1, 2] == 666

    assert x[0][0] == 111
    assert x[0][1] == 222
    assert x[0][2] == 333
    assert x[1][0] == 444
    assert x[1][1] == 555
    assert x[1][2] == 666

    # n-dim array (negative index)
    # en: get the n-th element of the n-dim array from the end
    # ja: n 次元配列の n 番目の要素を末尾から取得する
    x = np.array(
        [
            [111, 222, 333],
            [444, 555, 666],
        ]
    )
    assert x[-1, -1] == 666
    assert x[-1, -2] == 555
    assert x[-1, -3] == 444
    assert x[-2, -1] == 333
    assert x[-2, -2] == 222
    assert x[-2, -3] == 111

    # n-dim array slice
    # en: get a slice of the n-dim array
    # ja: n 次元配列のスライスを取得する
    x = np.array(
        [
            [111, 222, 333],
            [444, 555, 666],
        ]
    )
    assert (x[0:2, 0:2] == np.array([[111, 222], [444, 555]])).all()
    assert (x[0:2, 1:3] == np.array([[222, 333], [555, 666]])).all()
    assert (x[0:2, 0:3] == np.array([[111, 222, 333], [444, 555, 666]])).all()
    assert (x[0:2, 0:4] == np.array([[111, 222, 333], [444, 555, 666]])).all()
    assert (x[0:2, 0:0] == np.array([[], []])).all()
    assert (x[0:2, 0:-1] == np.array([[111, 222], [444, 555]])).all()
    assert (x[0:2, 0:-2] == np.array([[111], [444]])).all()
    assert (x[0:2, 0:-3] == np.array([[], []])).all()


def test_random_with_seed() -> None:
    """np.random"""
    # en: np.random and seed
    # ja: 乱数の生成とシード値の固定

    # without seed
    x1 = np.random.rand()
    x2 = np.random.rand()
    assert x1 != x2

    # with seed
    np.random.seed(0)
    x1 = np.random.rand()
    np.random.seed(0)
    x2 = np.random.rand()
    assert x1 == x2

    np.random.seed(0)
    assert np.random.rand(3) == pytest.approx(
        np.array([0.5488135, 0.71518937, 0.60276338])
    )

    np.random.seed(0)
    assert np.random.rand(2, 3) == pytest.approx(
        np.array(
            [
                [0.54881350, 0.71518937, 0.60276338],
                [0.54488318, 0.42365480, 0.64589411],
            ]
        )
    )
