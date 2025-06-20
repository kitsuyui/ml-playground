from .permutation_rank import permutation_rank


def test_permutation_rank() -> None:
    """
    Test the permutation_rank function.
    """
    items = ["A"]
    assert permutation_rank(["A"], items) == 0

    items = ["A", "B"]
    assert permutation_rank(["A", "B"], items) == 0
    assert permutation_rank(["B", "A"], items) == 1
    assert permutation_rank(["A"], items) == 0
    assert permutation_rank(["B"], items) == 1

    items = ["A", "B", "C"]
    assert permutation_rank(["A", "B", "C"], items) == 0
    assert permutation_rank(["A", "C", "B"], items) == 1
    assert permutation_rank(["B", "A", "C"], items) == 2
    assert permutation_rank(["B", "C", "A"], items) == 3
    assert permutation_rank(["C", "A", "B"], items) == 4
    assert permutation_rank(["C", "B", "A"], items) == 5
    assert permutation_rank(["A", "B"], items) == 0
    assert permutation_rank(["A", "C"], items) == 1
    assert permutation_rank(["B", "A"], items) == 2
    assert permutation_rank(["B", "C"], items) == 3
    assert permutation_rank(["C", "A"], items) == 4
    assert permutation_rank(["C", "B"], items) == 5
    assert permutation_rank(["A"], items) == 0
    assert permutation_rank(["B"], items) == 1
    assert permutation_rank(["C"], items) == 2

    items = ["A", "B", "C", "D"]
    assert permutation_rank(["A", "B", "C", "D"], items) == 0
    assert permutation_rank(["A", "B", "D", "C"], items) == 1
    assert permutation_rank(["A", "C", "B", "D"], items) == 2
    assert permutation_rank(["A", "C", "D", "B"], items) == 3
    assert permutation_rank(["A", "D", "B", "C"], items) == 4
    assert permutation_rank(["A", "D", "C", "B"], items) == 5
    assert permutation_rank(["B", "A", "C", "D"], items) == 6
    assert permutation_rank(["B", "A", "D", "C"], items) == 7
    assert permutation_rank(["B", "C", "A", "D"], items) == 8
    assert permutation_rank(["B", "C", "D", "A"], items) == 9
    assert permutation_rank(["B", "D", "A", "C"], items) == 10
    assert permutation_rank(["B", "D", "C", "A"], items) == 11
    assert permutation_rank(["C", "A", "B", "D"], items) == 12
    assert permutation_rank(["C", "A", "D", "B"], items) == 13
    assert permutation_rank(["C", "B", "A", "D"], items) == 14
    assert permutation_rank(["C", "B", "D", "A"], items) == 15
    assert permutation_rank(["C", "D", "A", "B"], items) == 16
    assert permutation_rank(["C", "D", "B", "A"], items) == 17
    assert permutation_rank(["D", "A", "B", "C"], items) == 18
    assert permutation_rank(["D", "A", "C", "B"], items) == 19
    assert permutation_rank(["D", "B", "A", "C"], items) == 20
    assert permutation_rank(["D", "B", "C", "A"], items) == 21
    assert permutation_rank(["D", "C", "A", "B"], items) == 22
    assert permutation_rank(["D", "C", "B", "A"], items) == 23

    items = range(18)
    assert permutation_rank([0, 1, 2], items) == 0
