.PHONY: help install-uv ensure-uv install format lint typecheck test test-fast test-cov test-property test-integration test-slow test-collect clean build pre-commit ci dev check notebook docs docs-serve

# Configuration
HELP_FORMAT_WIDTH = 17

help: ## Show this help message
	@echo "Available commands:"
	@awk -v width=$(HELP_FORMAT_WIDTH) 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-" width "s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-uv: ## Install uv using the official installer
	@echo "📦 Installing uv..."
	@if command -v uv > /dev/null 2>&1; then \
		echo "✅ uv is already installed! Use 'uv self update' to upgrade."; \
	else \
		if [ "$$(uname)" = "Darwin" ] || [ "$$(uname)" = "Linux" ]; then \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		elif [ "$$(uname)" = "MINGW64_NT" ] || [ "$$(uname)" = "MSYS_NT" ]; then \
			powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"; \
		else \
			echo "❌ Unsupported platform: $$(uname)"; \
			echo "Please visit https://docs.astral.sh/uv/getting-started/installation/ for manual installation"; \
			exit 1; \
		fi; \
		echo "✅ uv installation complete! Please restart your shell or run: source ~/.bashrc"; \
	fi

ensure-uv: ## Check if uv is installed
	@which uv > /dev/null || (echo "❌ uv is not installed. Run 'make install-uv' to install it automatically, or visit https://docs.astral.sh/uv/getting-started/installation/" && exit 1)
	@echo "✅ uv is available!"

install-dev: ensure-uv ## Install dependencies and setup development environment
	uv sync --dev
	@echo "✅ Development environment ready!"

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .
	@echo "✅ Code formatted!"

format-check: ## Check code formatting (CI only - doesn't modify files)
	uv run ruff format --check .
	@echo "✅ Formatting check complete!"

lint: ## Run linting checks
	uv run ruff check .
	@echo "✅ Linting complete!"

typecheck: ## Run type checking with mypy
	uv run mypy .
	@echo "✅ Type checking complete!"

test-fast: ## Run fast tests (excludes slow tests)
	uv run pytest -m "not slow" -v
	@echo "✅ Fast tests complete!"

test-slow: ## Run only slow tests
	uv run pytest -m slow -v
	@echo "✅ Slow tests complete!"

test: ## Run all tests
	uv run pytest -v
	@echo "✅ All tests complete!"

test-cov: ## Run tests with coverage report
	uv run pytest --cov=lssvm --cov=utils --cov-report=term-missing --cov-report=html
	@echo "✅ Coverage report generated!"

test-property: ## Run only property-based tests
	uv run pytest -m property -v
	@echo "✅ Property tests complete!"

test-integration: ## Run only integration tests
	uv run pytest -m integration -v
	@echo "✅ Integration tests complete!"

# Quality checks
pre-commit: format lint typecheck test ## Run all pre-commit checks
	@echo "✅ All pre-commit checks passed!"

ci: lint typecheck test ## Run all CI checks
	@echo "✅ All CI checks passed!"

build: ## Build the package
	uv build
	@echo "✅ Package built!"

clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -exec rm -rf {} +
	find . -name ".mypy_cache" -exec rm -rf {} +
	find . -name ".ruff_cache" -exec rm -rf {} +
	find . -name "dist" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"

notebook: ## Start Jupyter notebook server
	uv run jupyter notebook
	@echo "🚀 Jupyter notebook server started!"

check: format-check lint typecheck ## Run all code quality checks (CI)
	@echo "✅ Quality checks complete!"

test-collect: ## Show test discovery without running them
	uv run pytest --collect-only
	@echo "🔍 Test collection complete!"

docs: ## Build documentation (placeholder)
	@echo "📚 Documentation build not yet implemented"

docs-serve: ## Serve documentation locally (placeholder)
	@echo "📚 Documentation serve not yet implemented"
