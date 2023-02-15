from example.product import product


def test_product() -> None:
    assert product([1, 2, 3]) == 6
    assert product([1, 2, 3, 4]) == 24
    assert product([]) == 1
