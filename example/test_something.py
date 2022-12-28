import example


def test_example() -> None:
    assert 1 + 1 == 2


def test_example_foo() -> None:
    assert example.foo + "World!" == "Hello, World!"
