import numpy as np
import polars as pl


def test_dataframe() -> None:
    df = pl.from_dict(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
        }
    )

    assert df["a"].sum() == 6
    assert df.shape == (3, 3)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(df["a"], pl.Series)
    assert df["a"].dtype == pl.Int64
    assert df["b"].dtype == pl.Float64
    assert df["c"].dtype == pl.Utf8

    # to numpy
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    c = df["c"].to_numpy()

    # pitfall: requires pyarrow to be installed for this to work
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert isinstance(c, np.ndarray)
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert a.dtype == int
    assert b.dtype == float
