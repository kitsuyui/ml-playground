import functools
import operator


def product(iterable: list[int]) -> int:
    return functools.reduce(operator.mul, iterable, 1)
