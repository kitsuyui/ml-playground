"""Pick the maximum item from given items and decrease it by 1.
It will repeat until all items are less than or equal to 0.
This means every item will be picked at least once.
The result is the sequence of the keys of the items.

Example:
When the items are A: 4.2, B: 3.5, C: 2.8, D: 1.1
The result will be A, B, A, C, B, A, C, B, A, D, C, B, A, D

0: A: 4.2, B: 3.5, C: 2.8, D: 1.1 ... pick A
1: A: 3.2, B: 3.5, C: 2.8, D: 1.1 ... pick B
1: A: 3.2, B: 2.5, C: 2.8, D: 1.1 ... pick A
2: A: 3.2, B: 2.5, C: 1.8, D: 1.1 ... pick C
3: A: 2.2, B: 2.5, C: 1.8, D: 1.1 ... pick B
4: A: 2.2, B: 1.5, C: 1.8, D: 1.1 ... pick A
5: A: 1.2, B: 1.5, C: 1.8, D: 1.1 ... pick C
5: A: 1.2, B: 1.5, C: 0.8, D: 1.1 ... pick B
6: A: 0.2, B: 1.5, C: 0.8, D: 1.1 ... pick A
7: A: 0.2, B: 0.5, C: 0.8, D: 1.1 ... pick D
8: A: 0.2, B: 0.5, C: 0.8, D: 0.1 ... pick C
9: A: 0.2, B: 0.5, C: -0.2, D: 0.1 ... pick B
10: A: 0.2, B: -0.5, C: -0.2, D: 0.1 ... pick A
11: A: -0.8, B: -0.5, C: -0.2, D: 0.1 ... pick D
12: A: -0.8, B: -0.5, C: -0.2, D: -0.9 ... End
"""

import heapq


Items = list[tuple[str, float]]
Result = list[str]


def pick_max_steps_v1(items: Items) -> Result:
    """Most simple way to pick maximum value from given items.

    n is the number of items.
    m is the maximum value of items.

    O(m * n * log(n))
    Because of sorting in each iteration. Python's sort() and sorted() use Timsort algorithm. O(n * log(n))
    """
    items = items[:]
    result = []
    while any(v > 0 for _, v in items):  # m loop
        items = sorted(
            items, key=lambda x: x[1], reverse=True
        )  # n * log(n) loop
        key, value = items[0]
        result.append(key)
        items[0] = (key, value - 1)
    return result


def pick_max_steps_v2(items: Items) -> Result:
    """Optimized way to pick maximum value from given items.

    n is the number of items.
    m is the maximum value of items.

    O(m * n)
    """
    result = []
    while any(v > 0 for _, v in items):  # m loop
        max_key = max(items, key=lambda x: x[1])[0]  # n loop
        items = [(k, v - 1) if k == max_key else (k, v) for k, v in items]
        result.append(max_key)
    return result


def pick_max_steps_v3(items: Items) -> Result:
    """Optimized way to pick maximum value from given items.

    n is the number of items.
    m is the maximum value of items.

    O(n + m * log(n))
    Because of heapify in each iteration. Python's heapq.heapify() uses O(n) time.
    After that, heappop() and heappush() use O(log(n)) time.
    """
    result = []
    heap: list[tuple[float, str]] = [(-v, k) for k, v in items if v > 0]
    heapq.heapify(heap)  # n loop
    while heap:  # m loop
        max_value, max_key = heapq.heappop(heap)  # log(n) loop
        max_value = -max_value
        result.append(max_key)
        max_value -= 1
        if max_value > 0:
            heapq.heappush(heap, (-max_value, max_key))  # log(n) loop

    return result
