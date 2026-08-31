.PHONY: install lint format typecheck test check build clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest -m "l0 or l1"

check: lint typecheck test

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
