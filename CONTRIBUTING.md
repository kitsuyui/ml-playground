# Contributing

Thank you for taking the time to improve `ml-playground`.

This repository is a Python workspace for machine learning samples and small
package experiments. Keep changes focused on one topic and prefer examples that
remain easy to run in a clean checkout.

## Development Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Packages live under `packages/`. The root Poe tasks run the matching task for
each package.

## Checks

Run formatting before opening a pull request:

```sh
uv run poe format
```

Run lint and type checks:

```sh
uv run poe check
```

Run the test suite:

```sh
uv run poe test
```

Build all packages when changing packaging metadata, package layout, or release
inputs:

```sh
uv run poe build
```

## Pull Requests

Before opening a pull request, please make sure that:

- the change is focused on one topic;
- relevant checks pass locally;
- README or package documentation updates are included when behavior changes;
- new examples remain covered by tests or runnable notebook checks where
  practical.

When reporting a failing check, include the command you ran and the relevant
error output.
