# kitsuyui_ml.legacy

## What is this?

This package keeps older playground examples that still need runnable test
coverage while their long-term home is undecided.

It is not a stable public API. If an example becomes useful outside this
playground, move it into a non-legacy package under `packages/` before new
callers depend on it.

## Lifecycle

`kitsuyui_ml.legacy` is intentionally retained until each module has one of the
following outcomes:

- It is moved to a non-legacy package with matching tests and documentation.
- It remains here as a runnable example because no supported package needs it.
- It is removed after no tests, notebooks, or package code depend on it.

A module is ready to leave this package when the replacement or removal PR
records the chosen destination, keeps the relevant examples covered by tests,
and updates any notebooks or README references that still point here.

There is no scheduled package-wide removal date. Treat this package as a
holding area for examples to migrate or delete module by module, not as a place
for new reusable APIs.

# LICENSE

BSD 3-Clause License
