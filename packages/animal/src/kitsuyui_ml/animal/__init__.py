"""A package for animals.

This package provides a simple class `Animal` and a subclass `Dog` to represent animals.
This package is just for example.

Example:
    >>> import kitsuyui.animal
    >>> dog = kitsuyui.animal.Dog("Rex")
    >>> dog.speak()
    'Bark'
    >>> kitsuyui.animal.example()
    Bark
"""

import abc

# https://packaging-guide.openastronomy.org/en/latest/advanced/versioning.html
from ._version import __version__


class Animal:
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def speak(self):
        raise NotImplementedError(
            "Subclass must implement abstract method"
        )  # pragma: no cover


class Dog(Animal):
    def speak(self):
        return "Bark"


__all__ = [
    "__version__",
    "Animal",
    "Dog",
    "example",
]
