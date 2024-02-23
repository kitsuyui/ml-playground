# ml-playground

[![codecov](https://codecov.io/gh/kitsuyui/ml-playground/branch/main/graph/badge.svg?token=DW0671X9QF)](https://codecov.io/gh/kitsuyui/ml-playground)
[
![python-v3.9](https://img.shields.io/badge/python-v3.9-blue)
![python-v3.10](https://img.shields.io/badge/python-v3.10-blue)
![python-v3.11](https://img.shields.io/badge/python-v3.11-blue)
![python-v3.11](https://img.shields.io/badge/python-v3.12-blue)
](https://github.com/kitsuyui/ml-playground/actions/workflows/python-test.yml?query=branch%3Amain)

## en: What is this?

This is a repository for machine learning playground and sample codes.
I want to organize samples for myself because sample codes for machine learning are often of low quality.

- GitHub Actions runs tests for each version of Python (3.9 - 3.12) always.
  - Python 3.8, which is the LTS of Python, will be provided with security fixes until 2024. https://devguide.python.org/versions/
  - numpy has ended support for Python 3.8 due to NEP 29. https://numpy.org/neps/nep-0029-deprecation_policy.html 
- [Renovate](https://github.com/apps/renovate) continuously updates dependencies.
- [Codecov](https://app.codecov.io/gh/kitsuyui/ml-playground) measures test coverage.
- [nbmake](https://github.com/treebeardtech/nbmake) tests Jupyter ipynb files.

By doing so, I will ensure that sample codes always function as samples.

## ja: これは何？

機械学習の実験場・サンプルコードのリポジトリです。
機械学習周りのサンプルコードはクォリティが高くないことが多いので、自分用にサンプルを整理したいと考えています。

- GitHub Actions で常に各バージョンの Python (3.9 - 3.12) でテストが実行されます。
  - Python の　LTS である 3.8 は 2024 年まで security fix が提供されます。 https://devguide.python.org/versions/
  - numpy は NEP 29 により Python 3.8 のサポートを終了しています。 https://numpy.org/neps/nep-0029-deprecation_policy.html
- [Renovate](https://github.com/apps/renovate) が継続的に依存パッケージを更新します。
- [Codecov](https://app.codecov.io/gh/kitsuyui/ml-playground) がテストカバレッジを計測します。
- [nbmake](https://github.com/treebeardtech/nbmake) で Jupyter の ipynb ファイルもテストします。

これらによって、サンプルコードが常にサンプルとして機能することを保証していきます。

# Usage

## Install dependencies

Install dependencies with [poetry](https://python-poetry.org/).

```sh
poetry install
```

## Additional installation

### typos

This project uses typos.
Please install it according to https://github.com/crate-ci/typos#install

## Run poe tasks

## [Install poe as a poetry plugin](https://github.com/nat-n/poethepoet#installation)

```sh
poetry self add 'poethepoet[poetry_plugin]'
```

## Run tests

```sh
poetry poe test
```

## Format

```sh
poetry poe format
```

- [isort](https://pycqa.github.io/isort/) for import sorting
- [black](https://black.readthedocs.io/en/stable/) for formatting
- [pyupgrade](https://github.com/asottile/pyupgrade) for upgrading syntax to the latest version of Python

## Lint

```sh
poetry poe check
```

- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [flake8](https://flake8.pycqa.org/en/latest/) for linting
- [black](https://black.readthedocs.io/en/stable/) for formatting check
- [isort](https://pycqa.github.io/isort/) for import sorting check

# LICENSE

MIT License
