# Makefile for data_processing.py testing

.PHONY: help test test-unit test-integration test-fast test-all coverage clean install lint

# Default target
help:
	@echo "Available commands:"
	@echo "  make install          Install test dependencies"
	@echo "  make test            Run all tests with coverage"
	@echo "  make test-unit       Run only unit tests"
	@echo "  make test-integration Run only integration tests"
	@echo "  make test-fast       Run tests excluding slow ones"
	@echo "  make test-critical   Run only critical tests"
	@echo "  make coverage        Generate detailed coverage report"
	@echo "  make clean           Remove test artifacts"
	@echo "  make lint            Run linting checks"

# Install dependencies
install:
	pip install -r test-requirements.txt

# Run all tests
test:
	python run_tests.py

# Run unit tests only
test-unit:
	python run_tests.py --unit

# Run integration tests only
test-integration:
	python run_tests.py --integration

# Run fast tests (skip slow)
test-fast:
	python run_tests.py --fast

# Run critical tests only
test-critical:
	python run_tests.py --critical

# Generate coverage report
coverage:
	python run_tests.py --coverage
	@echo "Coverage report generated in htmlcov/index.html"

# Clean test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf interp/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Lint code (optional - requires flake8/black)
lint:
	@echo "Running linting checks..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 interp/data_processing.py tests/test_data_processing.py; \
	else \
		echo "flake8 not installed, skipping..."; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		black --check interp/data_processing.py tests/test_data_processing.py; \
	else \
		echo "black not installed, skipping..."; \
	fi

# Run specific test class
test-class:
	@read -p "Enter test class name (e.g., TestModelManager): " class_name; \
	python run_tests.py --specific $$class_name

# Watch for changes and rerun tests (requires pytest-watch)
watch:
	@if command -v ptw >/dev/null 2>&1; then \
		ptw -- -x tests/test_data_processing.py; \
	else \
		echo "pytest-watch not installed. Install with: pip install pytest-watch"; \
	fi

# Run tests with debugging
debug:
	python run_tests.py --pdb --lf -v

# Quick test run (no coverage, minimal output)
quick:
	python run_tests.py --no-cov --quiet --fast