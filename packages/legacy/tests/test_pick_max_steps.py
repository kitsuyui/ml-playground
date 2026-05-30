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


@pytest.mark.parametrize(
    "items",
    [
        [("A", 4.2), ("B", 3.5), ("C", 2.8), ("D", 1.1)],
        [("X", 1.0), ("Y", 1.0)],
        [("Z", 3.0)],
    ],
)
def test_all_versions_produce_identical_output(items):  # type: ignore[no-untyped-def]
    """v1, v2, and v3 must return the same sequence for inputs without mid-run ties."""
    r1 = pick_max_steps_v1(items)
    r2 = pick_max_steps_v2(items)
    r3 = pick_max_steps_v3(items)
    assert r1 == r3, f"v1 and v3 differ: {r1!r} != {r3!r}"
    assert r2 == r3, f"v2 and v3 differ: {r2!r} != {r3!r}"


def test_v1_tie_break_diverges_from_v2_v3_on_mid_run_ties() -> None:
    """v1 uses evolving list-order tie-breaking; v2 and v3 use original input order.

    When items that start with distinct values become equal mid-run, v1 picks
    whichever item happens to be earliest in the *current* (sorted) list, which
    drifts from original input order. v2/v3 always prefer the item that appeared
    first in the original input, so they agree with each other but not with v1.

    Example: [A=2, B=2, C=1]
      v1 → [A, B, B, A, C]  (after B is updated to 1, it precedes A in sorted order)
      v2 → [A, B, A, B, C]  (first-max always resolves to original insertion order)
      v3 → [A, B, A, B, C]  (heap stores original index; ties resolve by insertion order)
    """
    items = [("A", 2.0), ("B", 2.0), ("C", 1.0)]
    r1 = pick_max_steps_v1(items)
    r2 = pick_max_steps_v2(items)
    r3 = pick_max_steps_v3(items)
    assert r2 == r3, f"v2 and v3 must agree: {r2!r} != {r3!r}"
    assert r1 != r3, "v1 is expected to diverge from v3 on mid-run ties"
