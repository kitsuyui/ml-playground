# ml-playground

![Coverage](https://raw.githubusercontent.com/kitsuyui/octocov-central/main/badges/kitsuyui/ml-playground/coverage.svg)
[![type: ignore](https://raw.githubusercontent.com/kitsuyui/ml-playground/gh-counter-assets/badges/type-ignore.svg)](https://github.com/kitsuyui/ml-playground/search?q=%22type%3A+ignore%22+path%3Apackages&type=code)
[
![python-v3.10](https://img.shields.io/badge/python-v3.10-blue)
![python-v3.11](https://img.shields.io/badge/python-v3.11-blue)
![python-v3.12](https://img.shields.io/badge/python-v3.12-blue)
![python-v3.13](https://img.shields.io/badge/python-v3.13-blue)
](https://github.com/kitsuyui/ml-playground/actions/workflows/python-test.yml?query=branch%3Amain)

## en: What is this?

This is a repository for machine learning playground and sample codes.
I want to organize samples for myself because sample codes for machine learning are often of low quality.
The repository tracks Python `# type: ignore` markers with [`gh-counter`](https://github.com/kitsuyui/gh-counter).

- GitHub Actions runs tests for each versions of Python. (Currently 3.10 - 3.12)
  - This range is determined by lifecycle of Python versions and libraries.
  - Python support table: https://devguide.python.org/versions/
  - numpy support table: https://numpy.org/neps/nep-0029-deprecation_policy.html
- [Renovate](https://github.com/apps/renovate) continuously updates dependencies.
- [nbmake](https://github.com/treebeardtech/nbmake) tests Jupyter ipynb files.

By doing so, I will ensure that sample codes always function as samples.

## Project health

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
- Report suspected vulnerabilities through [SECURITY.md](SECURITY.md), not a
  public issue.
- Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) when participating in issues
  or pull requests.

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

## Development

This repository uses [lefthook](https://lefthook.dev/) to run the same checks as CI
locally, so problems surface before they reach CI.

```sh
# Install dependencies
uv sync

# Install the Git hooks (once; requires lefthook on your PATH)
lefthook install
```

Once installed, the hooks run automatically:

- **pre-commit**: `uv run poe check`
- **pre-push**: `uv run poe check` and `uv run poe test`

You can also run the checks manually:

```sh
uv run poe check
uv run poe test
```

CI still runs the full matrix (see `.github/workflows/`); the hooks only bring that
feedback earlier on your machine.

# LICENSE

MIT License
