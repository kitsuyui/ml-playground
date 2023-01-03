import numpy as np
import scipy.sparse as sps


def test_sparse_matrix() -> None:
    """Test sparse matrix.

    en: Example of sparse matrix.
    ja: 疎行列の扱いの見本
    """

    base_matrix = np.array(
        [
            [1, 0, 0, 4],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
        ]
    )
    assert base_matrix.shape == (3, 4)

    # CSR (Compressed Sparse Row)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    mat_csr = sps.csr_matrix(base_matrix)
    assert mat_csr.shape == (3, 4)
    assert (mat_csr.toarray() == base_matrix).all()
    # internal representation
    assert (mat_csr.data == np.array([1, 4, 2, 3])).all()
    assert (mat_csr.indices == np.array([0, 3, 1, 2])).all()
    assert (mat_csr.indptr == np.array([0, 2, 3, 4])).all()

    # CSC (Compressed Sparse Column)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
    mat_csc = sps.csc_matrix(base_matrix)
    assert mat_csc.shape == (3, 4)
    assert (mat_csc.toarray() == base_matrix).all()
    # internal representation
    assert (mat_csc.data == np.array([1, 2, 3, 4])).all()
    assert (mat_csc.indices == np.array([0, 1, 2, 0])).all()
    assert (mat_csc.indptr == np.array([0, 1, 2, 3, 4])).all()

    # COO (Coordinate)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    mat_coo = sps.coo_matrix(base_matrix)
    assert mat_coo.shape == (3, 4)
    assert (mat_coo.toarray() == base_matrix).all()
    # internal representation
    assert (mat_coo.data == np.array([1, 4, 2, 3])).all()
    assert (mat_coo.row == np.array([0, 0, 1, 2])).all()
    assert (mat_coo.col == np.array([0, 3, 1, 2])).all()

    # DIA (Diagonal)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html
    mat_dia = sps.dia_matrix(base_matrix)
    assert mat_dia.shape == (3, 4)
    assert (mat_dia.toarray() == base_matrix).all()
    # internal representation
    assert (mat_dia.data == np.array([[1, 2, 3, 0], [0, 0, 0, 4]])).all()
    assert (mat_dia.offsets == np.array([0, 3])).all()

    # BSR (Block Sparse Row)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
    mat_bsr = sps.bsr_matrix(base_matrix)
    assert mat_bsr.shape == (3, 4)
    assert (mat_bsr.toarray() == base_matrix).all()
    # internal representation
    assert (mat_bsr.data == np.array([[[1]], [[4]], [[2]], [[3]]])).all()
    # assert (mat_bsr.indices == np.array([0, 3, 1, 2])).all()
    # assert mat_bsr.indptr == np.array([0, 1, 2, 3, 4])
    assert (mat_bsr.blocksize == np.array([1, 1])).all()

    # DOK (Dictionary of Keys)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html
    mat_dok = sps.dok_matrix(base_matrix)
    assert mat_dok.shape == (3, 4)
    assert (mat_dok.toarray() == base_matrix).all()
    # internal representation
    assert list(mat_dok.keys()) == [(0, 0), (0, 3), (1, 1), (2, 2)]
    assert list(mat_dok.values()) == [1, 4, 2, 3]

    # LIL (Row-based list of list)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    mat_lil = sps.lil_matrix(base_matrix)
    assert mat_lil.shape == (3, 4)
    assert (mat_lil.toarray() == base_matrix).all()
    # internal representation
    assert list(mat_lil.rows) == [[0, 3], [1], [2]]
    assert list(mat_lil.data) == [[1, 4], [2], [3]]

    # en: The same operations as a normal matrix are supported
    # ja: 通常の行列と同じ操作をサポートしている
    # addition
    assert (
        (mat_csr + mat_csc).toarray()
        == np.array([[2, 0, 0, 8], [0, 4, 0, 0], [0, 0, 6, 0]])
    ).all()
    # multiplication
    assert (
        (mat_csr @ mat_csr.T).toarray()
        == np.array([[17, 0, 0], [0, 4, 0], [0, 0, 9]])
    ).all()
    # transpose
    print((mat_csr.T).toarray())
    assert (
        mat_csr.T.toarray()
        == np.array(
            [
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
                [4, 0, 0],
            ]
        )
    ).all()


def test_sparse_vector() -> None:
    """Example of sparse vector."""

    base_vector = np.array([1, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 3])

    # CSR (Compressed Sparse Row)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
    vec_csr = sps.csr_array(base_vector)
    assert vec_csr.shape == (1, 12)
    assert (vec_csr.toarray() == base_vector).all()
    # internal representation
    assert (vec_csr.data == np.array([1, 4, 2, 3])).all()
    assert (vec_csr.indices == np.array([0, 3, 5, 11])).all()

    # CSC (Compressed Sparse Column)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
    vec_csc = sps.csc_array(base_vector)
    assert vec_csc.shape == (1, 12)
    assert (vec_csc.toarray() == base_vector).all()
    # internal representation
    assert (vec_csc.data == np.array([1, 4, 2, 3])).all()
    assert (vec_csc.indices == np.array([0, 0, 0, 0])).all()

    # COO (Coordinate)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array
    vec_coo = sps.coo_array(base_vector)
    assert vec_coo.shape == (1, 12)
    assert (vec_coo.toarray() == base_vector).all()
    # internal representation
    assert (vec_coo.data == np.array([1, 4, 2, 3])).all()
    assert (vec_coo.row == np.array([0, 0, 0, 0])).all()
    assert (vec_coo.col == np.array([0, 3, 5, 11])).all()

    # DIA (Diagonal)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.html#scipy.sparse.dia_array
    vec_dia = sps.dia_array(base_vector)
    assert vec_dia.shape == (1, 12)
    assert (vec_dia.toarray() == base_vector).all()

    # BSR (Block Sparse Row)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.html#scipy.sparse.bsr_array
    vec_bsr = sps.bsr_array(base_vector)
    assert vec_bsr.shape == (1, 12)
    assert (vec_bsr.toarray() == base_vector).all()
