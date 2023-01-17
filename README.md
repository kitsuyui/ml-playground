# ml-playground

[![codecov](https://codecov.io/gh/kitsuyui/ml-playground/branch/main/graph/badge.svg?token=DW0671X9QF)](https://codecov.io/gh/kitsuyui/ml-playground)
[
![python-v3.8](https://img.shields.io/badge/python-v3.8-blue)
![python-v3.9](https://img.shields.io/badge/python-v3.9-blue)
![python-v3.10](https://img.shields.io/badge/python-v3.10-blue)
![python-v3.11](https://img.shields.io/badge/python-v3.11-blue)
](https://github.com/kitsuyui/ml-playground/actions/workflows/python-test.yml?query=branch%3Amain)

## en: What is this?

This is a repository for machine learning playground and sample codes.
I want to organize samples for myself because sample codes for machine learning are often of low quality.

- GitHub Actions runs tests for each version of Python (3.8 - 3.11) always.
- [Renovate](https://github.com/apps/renovate) continuously updates dependencies.
- [Codecov](https://app.codecov.io/gh/kitsuyui/ml-playground) measures test coverage.
- [nbmake](https://github.com/treebeardtech/nbmake) tests Jupyter ipynb files.

By doing so, I will ensure that sample codes always function as samples.

## ja: これは何？

機械学習の実験場・サンプルコードのリポジトリです。
機械学習周りのサンプルコードはクォリティが高くないことが多いので、自分用にサンプルを整理したいと考えています。

- GitHub Actions で常に各バージョンの Python (3.8 - 3.11) でテストが実行されます。
- [Renovate](https://github.com/apps/renovate) が継続的に依存パッケージを更新します。
- [Codecov](https://app.codecov.io/gh/kitsuyui/ml-playground) がテストカバレッジを計測します。
- [nbmake](https://github.com/treebeardtech/nbmake) で Jupyter の ipynb ファイルもテストします。

これらによって、サンプルコードが常にサンプルとして機能することを保証していきます。

# Usage

## Install dependencies

Install dependencies with [poetry](https://python-poetry.org/).

```bash
poetry install
```

## Run tests

```bash
poetry poe test
```

## Format

```bash
poetry poe format
```

- [isort](https://pycqa.github.io/isort/) for import sorting
- [black](https://black.readthedocs.io/en/stable/) for formatting
- [pyupgrade](https://github.com/asottile/pyupgrade) for upgrading syntax to the latest version of Python

## Lint

```bash
poetry poe check
```

- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [flake8](https://flake8.pycqa.org/en/latest/) for linting
- [black](https://black.readthedocs.io/en/stable/) for formatting check
- [isort](https://pycqa.github.io/isort/) for import sorting check

# LICENSE

MIT License
