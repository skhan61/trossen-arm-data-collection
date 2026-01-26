.PHONY: install test lint format clean pre-commit push help

help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linter (ruff check)"
	@echo "  format      - Format code (ruff format)"
	@echo "  clean       - Clean cache files"
	@echo "  pre-commit  - Install pre-commit hooks"
	@echo "  push        - Push to remote"

install:
	uv sync

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/ _logs/ 2>/dev/null || true

pre-commit:
	uv run pre-commit install

push:
	git add -A
	git commit -m "Update codebase" || true
	git push origin HEAD
