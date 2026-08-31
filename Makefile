.PHONY: install develop sync rust-fmt rust-lint rust-test lint fmt typecheck test check build clean

install:
	pip install -e ".[dev]"

develop:
	maturin develop

# Push the current branch and print the one-liner to run the GPU CI on Colab.
sync:
	@git push -u origin $$(git branch --show-current)
	@echo
	@echo "Open notebooks/dev.ipynb on Colab, or paste into a GPU cell:"
	@echo "  !curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null && . \$$HOME/.cargo/env \\"
	@echo "   && git clone --depth 1 -b $$(git branch --show-current) https://github.com/mansoor-mamnoon/caliper \\"
	@echo "   && cd caliper && pip -q install -e '.[dev]' && cargo test --all && pytest -q && caliper doctor"

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
