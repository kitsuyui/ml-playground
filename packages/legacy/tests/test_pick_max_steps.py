import pytest

from kitsuyui_ml.legacy.algorithms.pick_max_steps import (
    DEFAULT_MAX_RESULT_LENGTH,
    pick_max_steps_v1,
    pick_max_steps_v2,
    pick_max_steps_v3,
)


def test_pick_sequence_v1() -> None:
    items = [
        ("A", 4.2),
        ("B", 3.5),
        ("C", 2.8),
        ("D", 1.1),
    ]
    tobe = [
        "A",
        "B",
        "A",
        "C",
        "B",
        "A",
        "C",
        "B",
        "A",
        "D",
        "C",
        "B",
        "A",
        "D",
    ]
    sequence = pick_max_steps_v1(items)
    assert sequence == tobe


def test_pick_sequence_v2() -> None:
    items = [
        ("A", 4.2),
        ("B", 3.5),
        ("C", 2.8),
        ("D", 1.1),
    ]
    tobe = [
        "A",
        "B",
        "A",
        "C",
        "B",
        "A",
        "C",
        "B",
        "A",
        "D",
        "C",
        "B",
        "A",
        "D",
    ]
    sequence = pick_max_steps_v2(items)
    assert sequence == tobe


def test_pick_sequence_v3() -> None:
    items = [
        ("A", 4.2),
        ("B", 3.5),
        ("C", 2.8),
        ("D", 1.1),
    ]
    tobe = [
        "A",
        "B",
        "A",
        "C",
        "B",
        "A",
        "C",
        "B",
        "A",
        "D",
        "C",
        "B",
        "A",
        "D",
    ]
    sequence = pick_max_steps_v3(items)
    assert sequence == tobe


@pytest.mark.parametrize(
    "fn",
    [pick_max_steps_v1, pick_max_steps_v2, pick_max_steps_v3],
)
def test_tiebreak_preserves_input_order(fn):  # type: ignore[no-untyped-def]
    """All three implementations must pick earlier-input item first on equal values."""
    items = [("B", 1.0), ("A", 1.0)]
    result = fn(items)
    assert result == ["B", "A"], f"{fn.__name__} tiebreak differs: {result}"


@pytest.mark.parametrize(
    "fn",
    [pick_max_steps_v1, pick_max_steps_v2, pick_max_steps_v3],
)
def test_rejects_results_over_default_limit(fn):  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="more than"):
        fn([("A", DEFAULT_MAX_RESULT_LENGTH + 1.0)])


@pytest.mark.parametrize(
    "fn",
    [pick_max_steps_v1, pick_max_steps_v2, pick_max_steps_v3],
)
def test_rejects_results_over_custom_limit(fn):  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="more than"):
        fn([("A", 3.1)], max_result_length=3)


@pytest.mark.parametrize(
    "fn",
    [pick_max_steps_v1, pick_max_steps_v2, pick_max_steps_v3],
)
def test_rejects_infinite_positive_values(fn):  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="finite"):
        fn([("A", float("inf"))])
