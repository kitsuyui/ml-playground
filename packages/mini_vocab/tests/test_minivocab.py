import pytest

from kitsuyui_ml.mini_vocab import Vocab, build_vocab


def test_mini_vocab() -> None:
    """
    Test the Vocab class.
    """
    vocab = Vocab.create(["hello", "world"])
    assert len(vocab) == 2
    assert vocab.stoi["hello"] == 0
    assert vocab.itos[0] == "hello"
    assert vocab.stoi["world"] == 1
    assert vocab.itos[1] == "world"

    vocab.add_word("new_word")
    assert len(vocab) == 3
    assert vocab.stoi["new_word"] == 2
    assert vocab.itos[2] == "new_word"

    vocab.add_words(["another", "word"])
    assert len(vocab) == 5
    assert vocab.stoi["another"] == 3
    assert vocab.itos[3] == "another"
    assert vocab.stoi["word"] == 4
    assert vocab.itos[4] == "word"


def test_build_vocab() -> None:
    """
    Test the build_vocab function.
    """
    texts = ["hello world", "hello again", "world of python"]

    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    vocab = build_vocab(texts, simple_tokenizer, specials=["<pad>", "<unk>"])

    assert len(vocab) == 7  # 5 unique words + 2 special tokens
    assert vocab.stoi["<pad>"] == 0
    assert vocab.itos[0] == "<pad>"
    assert vocab.stoi["<unk>"] == 1
    assert vocab.itos[1] == "<unk>"
    assert vocab.stoi["hello"] == 2
    assert vocab.itos[2] == "hello"
    assert vocab.stoi["world"] == 3
    assert vocab.itos[3] == "world"
    assert vocab.stoi["again"] == 4
    assert vocab.itos[4] == "again"
    assert vocab.stoi["of"] == 5
    assert vocab.itos[5] == "of"
    assert vocab.stoi["python"] == 6
    assert vocab.itos[6] == "python"

    # when specials are not provided

    vocab_no_specials = build_vocab(texts, simple_tokenizer)
    assert len(vocab_no_specials) == 5  # 5 unique words
    assert vocab_no_specials.stoi["hello"] == 0
    assert vocab_no_specials.itos[0] == "hello"
    assert vocab_no_specials.stoi["world"] == 1
    assert vocab_no_specials.itos[1] == "world"
    assert vocab_no_specials.stoi["again"] == 2
    assert vocab_no_specials.itos[2] == "again"
    assert vocab_no_specials.stoi["of"] == 3
    assert vocab_no_specials.itos[3] == "of"
    assert vocab_no_specials.stoi["python"] == 4
    assert vocab_no_specials.itos[4] == "python"


def test_build_vocab_uses_first_seen_order_not_frequency() -> None:
    """
    Test that build_vocab assigns indices by first-seen token order.
    """
    texts = [
        "rare common common",
        "common later",
    ]

    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    vocab = build_vocab(texts, simple_tokenizer, specials=["<pad>"])

    assert vocab.stoi["<pad>"] == 0
    assert vocab.stoi["rare"] == 1
    assert vocab.stoi["common"] == 2
    assert vocab.stoi["later"] == 3


def test_vocab_max_size_limits_unique_words() -> None:
    """
    Test that max_size rejects new unique words after the bound is reached.
    """
    vocab = Vocab.create(["hello", "world"], max_size=2)

    vocab.add_word("hello")

    assert len(vocab) == 2
    with pytest.raises(ValueError, match="max_size exceeded"):
        vocab.add_word("again")


def test_vocab_rejects_negative_max_size() -> None:
    """
    Test that negative max_size is rejected before adding words.
    """
    with pytest.raises(ValueError, match="max_size must be non-negative"):
        Vocab(max_size=-1)


def test_build_vocab_max_size_limits_tokens_and_specials() -> None:
    """
    Test that build_vocab applies max_size to specials and regular tokens.
    """
    texts = [
        "rare common common",
        "common later",
    ]

    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    vocab = build_vocab(
        texts, simple_tokenizer, specials=["<pad>"], max_size=4
    )

    assert len(vocab) == 4
    assert vocab.stoi["<pad>"] == 0
    assert vocab.stoi["rare"] == 1
    assert vocab.stoi["common"] == 2
    assert vocab.stoi["later"] == 3

    with pytest.raises(ValueError, match="max_size exceeded"):
        build_vocab(texts, simple_tokenizer, specials=["<pad>"], max_size=3)
