.PHONY: install develop rust-fmt rust-lint rust-test lint fmt typecheck test check build clean

install:
	pip install -e ".[dev]"

develop:
	maturin develop

rust-fmt:
	cargo fmt --all --check

rust-lint:
	cargo clippy --all-targets --all-features -- -D warnings

rust-test:
	cargo test --all

lint:
	ruff check .
	ruff format --check .

fmt:
	cargo fmt --all
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest -m "l0 or l1"

check: rust-fmt rust-lint rust-test lint typecheck test

build:
	maturin build --release

clean:
	rm -rf target build dist .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
