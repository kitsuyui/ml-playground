"""
This package provides Vocab implementation similar to `torchtext.vocab.Vacab`

torchtext is a convenient library, but its development has ended.
spacy Vocab is a good alternative. but not easy to use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import MappingProxyType

# https://packaging-guide.openastronomy.org/en/latest/advanced/versioning.html
from ._version import __version__


class Vocab:
    """
    A simple vocabulary class that maps words to indices and vice versa.
    It is similar to `torchtext.vocab.Vocab`
    but does not depend on torch and torchtext.
    """

    __slots__ = ("_current_index", "_itos", "_max_size", "_stoi")

    def __init__(self, *, max_size: int | None = None) -> None:
        if max_size is not None and max_size < 0:
            msg = "max_size must be non-negative"
            raise ValueError(msg)
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        self._current_index: int = 0
        self._max_size = max_size

    @property
    def stoi(self) -> MappingProxyType[str, int]:
        return MappingProxyType(self._stoi)

    @property
    def itos(self) -> MappingProxyType[int, str]:
        return MappingProxyType(self._itos)

    @property
    def current_index(self) -> int:
        return self._current_index

    def add_word(self, word: str) -> None:
        """
        Add a single word to the vocabulary.
        """
        if word in self._stoi:
            return
        if self._max_size is not None and len(self._stoi) >= self._max_size:
            msg = "max_size exceeded"
            raise ValueError(msg)
        self._stoi[word] = self._current_index
        self._itos[self._current_index] = word
        self._current_index += 1

    def add_words(self, words: Iterable[str]) -> None:
        """
        Add multiple words to the vocabulary.
        """
        for word in words:
            self.add_word(word)

    def __len__(self) -> int:
        return len(self._stoi)

    @classmethod
    def create(
        cls, words: Iterable[str], *, max_size: int | None = None
    ) -> Vocab:
        """
        Create a vocabulary from a list of words.
        """
        vocab = cls(max_size=max_size)
        vocab.add_words(words)
        return vocab


def build_vocab(
    texts: list[str],
    tokenizer: Callable[[str], list[str]],
    *,
    specials: list[str] | None = None,
    max_size: int | None = None,
) -> Vocab:
    """
    Build a vocabulary from a list of texts using a tokenizer.
    When `specials` is provided, it will be added to the vocabulary.
    This is useful for adding special tokens like
    `<unknown>`, `<pad>`, `<bos>`, and `<eos>`.
    Vocabulary indices are deterministic: special tokens are inserted first
    in the provided order, then regular tokens are inserted in first-seen
    order across `texts`. Token frequency does not affect index assignment.
    `max_size` limits the number of unique tokens in the vocabulary.
    """
    vocab = Vocab(max_size=max_size)
    if specials is not None:
        vocab.add_words(specials)
    for text in texts:
        vocab.add_words(tokenizer(text))
    return vocab


__all__ = [
    "Vocab",
    "__version__",
    "build_vocab",
]
