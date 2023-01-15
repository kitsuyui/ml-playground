import io
import pathlib
from typing import Any

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

BASE_PATH = pathlib.Path("data/examples/graphs")
BASE_PATH.mkdir(parents=True, exist_ok=True)
SnapShot = Any


def test_matplotlib(snapshot: SnapShot) -> None:
    """Test matplotlib svg output."""
    snapshot.snapshot_dir = "snapshots/matplotlib"
    mpl.rcParams["svg.hashsalt"] = "test"
    fp = io.StringIO()

    _, ax = plt.subplots()
    x = np.linspace(0, 20, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("This is a plot")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    # plt.show()

    metadata = {"Date": None}  # Remove date from metadata for reproducibility
    plt.savefig(fp, format="svg", metadata=metadata)
    plt.savefig(BASE_PATH / "test1.svg", metadata=metadata)
    assert fp.getvalue() == snapshot


def test_matplotlib_hist(snapshot: SnapShot) -> None:
    """Test matplotlib histogram svg output."""
    snapshot.snapshot_dir = "snapshots/matplotlib"
    mpl.rcParams["svg.hashsalt"] = "test"
    fp = io.StringIO()

    _, ax = plt.subplots()
    np.random.seed(0)
    x = np.random.normal(size=1000)
    ax.hist(x, bins=20)
    ax.set_title("This is a histogram")
    ax.set_xlabel("x")
    ax.set_ylabel("y (random)")
    # plt.show()

    metadata = {"Date": None}  # Remove date from metadata for reproducibility
    plt.savefig(fp, format="svg", metadata=metadata)
    plt.savefig(BASE_PATH / "test2.svg", metadata=metadata)
    assert fp.getvalue() == snapshot


def test_matplotlib_bar(snapshot: SnapShot) -> None:
    """Test matplotlib bar svg output."""
    snapshot.snapshot_dir = "snapshots/matplotlib"
    mpl.rcParams["svg.hashsalt"] = "test"
    fp = io.StringIO()

    _, ax = plt.subplots()
    x = np.arange(5)
    y = np.arange(5)
    ax.bar(x, y)
    ax.set_title("This is a bar")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # plt.show()

    metadata = {"Date": None}  # Remove date from metadata for reproducibility
    plt.savefig(fp, format="svg", metadata=metadata)
    plt.savefig(BASE_PATH / "test3.svg", metadata=metadata)
    assert fp.getvalue() == snapshot


def test_matplotlib_scatter(snapshot: SnapShot) -> None:
    """Test matplotlib scatter svg output."""
    snapshot.snapshot_dir = "snapshots/matplotlib"
    mpl.rcParams["svg.hashsalt"] = "test"
    fp = io.StringIO()

    _, ax = plt.subplots()
    x = np.arange(5)
    y = np.arange(5)
    ax.scatter(x, y)
    ax.set_title("This is a scatter")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # plt.show()

    metadata = {"Date": None}  # Remove date from metadata for reproducibility
    plt.savefig(fp, format="svg", metadata=metadata)
    plt.savefig(BASE_PATH / "test4.svg", metadata=metadata)
    assert fp.getvalue() == snapshot
