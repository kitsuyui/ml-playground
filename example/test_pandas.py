import numpy as np
import pandas as pd  # type: ignore


def test_dataframe() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
        }
    )
    assert df["a"].sum() == 6
    assert df.shape == (3, 3)
    assert type(df) == pd.DataFrame
    assert type(df["a"]) == pd.Series
    assert df["a"].dtype == int
    assert df["b"].dtype == float
    assert df["c"].dtype == object

    # to numpy
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    c = df["c"].to_numpy()
    assert type(a) == np.ndarray
    assert type(b) == np.ndarray
    assert type(c) == np.ndarray
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)
    assert a.dtype == int
    assert b.dtype == float
    assert c.dtype == object

    # iloc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
    # Purely integer-location based indexing for selection by position.
    assert df.iloc[0, 0] == 1
    assert df.iloc[0, 1] == 4.0
    assert df.iloc[0, 2] == "a"
    assert df.iloc[1, 0] == 2
    assert df.iloc[1, 1] == 5.0
    assert df.iloc[1, 2] == "b"
    assert df.iloc[2, 0] == 3
    assert df.iloc[2, 1] == 6.0
    assert df.iloc[2, 2] == "c"
    # with slice
    assert df.iloc[0:1, 0].to_numpy().tolist() == [1]

    # loc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
    # Access a group of rows and columns by label(s) or a boolean array.
    assert df.loc[0, "a"] == 1
    assert df.loc[0, "b"] == 4.0
    assert df.loc[0, "c"] == "a"
    assert df.loc[1, "a"] == 2
    assert df.loc[1, "b"] == 5.0
    assert df.loc[1, "c"] == "b"
    assert df.loc[2, "a"] == 3
    assert df.loc[2, "b"] == 6.0
    assert df.loc[2, "c"] == "c"
    # with slice
    assert df.loc[0:1, "a"].to_numpy().tolist() == [1, 2]
