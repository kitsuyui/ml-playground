# kitsuyui_ml.mini_vocab

## What is this?

torchtext is a convenient library, but its development has ended.
This module provides a small alternative to `torchtext.vocab.Vocab`.
It provides a simple implementation in pure Python, not dependent on torch.
It is not intended to be used with large datasets, but rather as a simple implementation.

## Installation

This is not published to PyPI, so you need to install it from github.
You can also install it via local clone.

### Install via pip

```sh
$ pip install 'git+https://github.com/kitsuyui/ml-playground.git#egg=kitsuyui_ml.mini_vocab&subdirectory=packages/mini_vocab'
```

### Install via uv

```sh
$ uv add 'git+https://github.com/kitsuyui/ml-playground.git#egg=kitsuyui_ml.mini_vocab&subdirectory=packages/mini_vocab'
```

# LICENSE

BSD 3-Clause License
