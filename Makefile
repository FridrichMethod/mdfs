# mdfs developer tasks. Override the interpreter with `make PYTHON=... <target>`.
PYTHON ?= .venv/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit
PRECOMMIT := $(PYTHON) -m pre_commit

.DEFAULT_GOAL := help

.PHONY: help venv install test cov slow-test lint format mypy bandit precommit clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## Create a CUDA-enabled .venv (Python 3.12)
	uv venv .venv --python 3.12
	uv pip install -p .venv/bin/python -e ".[dev,mypy]" "jax[cuda12]"

install:  ## Editable install with dev + mypy extras
	uv pip install -p $(PYTHON) -e ".[dev,mypy]"

test:  ## Run the fast test suite
	$(PYTEST) -m "not slow and not gpu"

cov:  ## Run tests with coverage
	$(PYTEST) -m "not slow and not gpu" --cov=src/mdfs --cov-report=term-missing

slow-test:  ## Run the full suite including slow/e2e tests
	$(PYTEST)

lint:  ## Run ruff lint + format check
	$(RUFF) check src tests
	$(RUFF) format --check src tests

format:  ## Auto-format and fix lint
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

mypy:  ## Type-check the package
	$(MYPY) src/mdfs

bandit:  ## Security scan
	$(BANDIT) -r src/mdfs -c pyproject.toml -q -ll

precommit:  ## Run all pre-commit hooks (pre-commit + pre-push stages)
	$(PRECOMMIT) run --all-files --hook-stage pre-commit
	$(PRECOMMIT) run --all-files --hook-stage pre-push

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
