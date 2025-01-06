from .pick_max_steps import (
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
