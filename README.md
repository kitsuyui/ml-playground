# ml-playground

![Coverage](https://raw.githubusercontent.com/kitsuyui/octocov-central/main/badges/kitsuyui/ml-playground/coverage.svg)
[
![python-v3.10](https://img.shields.io/badge/python-v3.10-blue)
![python-v3.11](https://img.shields.io/badge/python-v3.11-blue)
![python-v3.12](https://img.shields.io/badge/python-v3.12-blue)
![python-v3.13](https://img.shields.io/badge/python-v3.13-blue)
](https://github.com/kitsuyui/ml-playground/actions/workflows/python-test.yml?query=branch%3Amain)

## en: What is this?

This is a repository for machine learning playground and sample codes.
I want to organize samples for myself because sample codes for machine learning are often of low quality.

- GitHub Actions runs tests for each versions of Python. (Currently 3.10 - 3.12)
  - This range is determined by lifecycle of Python versions and libraries.
  - Python support table: https://devguide.python.org/versions/
  - numpy support table: https://numpy.org/neps/nep-0029-deprecation_policy.html 
- [Renovate](https://github.com/apps/renovate) continuously updates dependencies.
- [Codecov](https://app.codecov.io/gh/kitsuyui/ml-playground) measures test coverage.
- [nbmake](https://github.com/treebeardtech/nbmake) tests Jupyter ipynb files.

By doing so, I will ensure that sample codes always function as samples.

# Usage

## Install dependencies

Install dependencies with [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Additional installation

### typos

This project uses typos.
Please install it according to https://github.com/crate-ci/typos#install

## Run poe tasks

## Run tests

```sh
uv run poe test
```

## Format

```sh
uv run poe format
```

## Lint

```sh
uv run poe check
```

# LICENSE

MIT License
