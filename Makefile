.PHONY: lint
lint: flake8 mypy check_import_order check_format

.PHONY: test
test: pytest test_types

.PHONY: format
format: isort black pyupgrade

.PHONY: isort
isort:
	isort example

.PHONY: black
black:
	black example

.PHONY: flake8
flake8:
	flake8 example

.PHONY: pyupgrade
pyupgrade:
	pyupgrade --py37-plus example/*.py

.PHONY: mypy
mypy:
	mypy example

.PHONY: check_import_order
check_import_order:
	isort --check-only --diff example

.PHONY: check_format
check_format:
	black --check example

.PHONY: test_types
test_types:
	mypy example

.PHONY: pytest
pytest:
	pytest --cov=example example --doctest-modules --cov-report=xml
