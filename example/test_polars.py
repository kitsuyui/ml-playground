import polars as pl
import pytest


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
    assert type(df) == pl.DataFrame
    assert type(df["a"]) == pl.Series
    assert df["a"].dtype == pl.Int64
    assert df["b"].dtype == pl.Float64
    assert df["c"].dtype == pl.Utf8

    # to numpy
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    with pytest.raises(NotImplementedError) as e:
        df["c"].to_numpy()
    assert (
        "Conversion of polars data type Utf8 to C-type not implemented."
        in str(e.value)
    )

    # pitfall: polars SeriesView is not a numpy array
    assert type(a) == pl.internals.series._numpy.SeriesView
    assert type(b) == pl.internals.series._numpy.SeriesView
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert a.dtype == int
    assert b.dtype == float
