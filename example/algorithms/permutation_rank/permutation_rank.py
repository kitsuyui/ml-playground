import math
from typing import TypeVar, List

T = TypeVar("T")


class FenwickTree:
    """
    Fenwick (Binary Indexed) Tree for frequency counting.
    Supports prefix sums and point updates in O(log n).
    """

    def __init__(self, size: int) -> None:
        self.n = size
        self.tree = [0] * (size + 1)

    def build(self, arr: List[int]) -> None:
        """
        Build the tree from initial array arr of length n in O(n).
        """
        for i, v in enumerate(arr):
            self.tree[i + 1] = v
        for i in range(1, self.n + 1):
            j = i + (i & -i)
            if j <= self.n:
                self.tree[j] += self.tree[i]

    def sum(self, idx: int) -> int:
        """
        Return sum of arr[0..idx] in O(log n).
        """
        s = 0
        i = idx + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def add(self, idx: int, delta: int) -> None:
        """
        Increment arr[idx] by delta in O(log n).
        """
        i = idx + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i


def permutation_rank(a: List[T], items: List[T]) -> int:
    """
    Compute the 0-based rank of the partial permutation a (length m) among
    all P(n, m) permutations of items (length n). Runs in O(m log n) time.

    Parameters:
    - a: list of m distinct items to be ranked
    - items: sorted list of n unique items used as the full set

    Returns:
    - rank: 0-based index of a in lexicographic order of P(n, m)

    Example:
        items = ['A', 'B', 'C']
        permutation_rank(['B','C','A'], items)  # => 3
    """
    n = len(items)
    m = len(a)
    # Map each item to its index in the sorted items list
    index_map = {item: i for i, item in enumerate(items)}

    fact = math.factorial
    denom = fact(n - m)

    # Initialize Fenwick Tree with all 1s (each item available)
    bit = FenwickTree(n)
    bit.build([1] * n)

    rank = 0
    for i, ai in enumerate(a):
        k = index_map[ai]
        # count of remaining items less than ai
        c = bit.sum(k - 1) if k > 0 else 0
        # number of permutations for each choice at this position
        block = fact(n - i - 1) // denom
        rank += c * block
        # mark ai as used
        bit.add(k, -1)

    return rank
