"""
This package provides Vocab implementation similar to `torchtext.vocab.Vacab`

torchtext is a convenient library, but its development has ended.
spacy Vocab is a good alternative. but not easy to use.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable
from collections import Counter

# https://packaging-guide.openastronomy.org/en/latest/advanced/versioning.html
from ._version import __version__


@dataclasses.dataclass
class Vocab:
    """
    A simple vocabulary class that maps words to indices and vice versa.
    It is similar to `torchtext.vocab.Vocab` but does not depend on torch and torchtext.
    """

    stoi: dict[str, int] = dataclasses.field(default_factory=dict)
    itos: dict[int, str] = dataclasses.field(default_factory=dict)
    current_index: int = 0

    def add_word(self, word: str) -> None:
        """
        Add a single word to the vocabulary.
        """
        if word not in self.stoi:
            self.stoi[word] = self.current_index
            self.itos[self.current_index] = word
            self.current_index += 1

    def add_words(self, words: Iterable[str]) -> None:
        """
        Add multiple words to the vocabulary.
        """
        for word in words:
            self.add_word(word)

    def __len__(self) -> int:
        return len(self.stoi)

    @classmethod
    def create(cls, words: Iterable[str]) -> Vocab:
        """
        Create a vocabulary from a list of words.
        """
        vocab = cls()
        vocab.add_words(words)
        return vocab


def build_vocab(
    texts: list[str],
    tokenizer: Callable[[str], list[str]],
    *,
    specials: list[str] | None = None,
) -> Vocab:
    """
    Build a vocabulary from a list of texts using a tokenizer.
    When `specials` is provided, it will be added to the vocabulary.
    This is useful for adding special tokens like `<unknown>`, `<pad>`, `<bos>`, `<eos>`.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    if specials is None:
        specials = []
    vocab = Vocab.create(words=specials)
    vocab.add_words(counter.keys())
    return vocab


__all__ = [
    "__version__",
    "Vocab",
    "build_vocab",
]
