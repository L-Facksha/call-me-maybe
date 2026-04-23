.PHONY: install run debug clean lint lint-strict help

help:
	@echo "Available targets:"
	@echo "  install     - Install project dependencies"
	@echo "  run         - Run the function calling system"
	@echo "  debug       - Run in debug mode"
	@echo "  clean       - Remove cache and temporary files"
	@echo "  lint        - Run flake8 and mypy checks"
	@echo "  lint-strict - Run mypy with strict mode"
	@echo "  test        - Run pytest tests"

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name .venv -prune -o -type d -name venv -prune
	rm -rf build/ dist/ *.egg-info/

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -v

.DEFAULT_GOAL := help
