import functools
import operator
from typing import List


def product(iterable: List[int]) -> int:
    return functools.reduce(operator.mul, iterable, 1)
