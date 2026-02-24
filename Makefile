.PHONY: install install-dev test lint format fetch-data train nowcast dashboard clean

# Install production dependencies
install:
	pip install -e .

# Install with dev dependencies
install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Lint code
lint:
	ruff check src/ tests/ scripts/
	mypy src/

# Format code
format:
	black src/ tests/ scripts/ notebooks/
	ruff check --fix src/ tests/ scripts/

# Fetch latest FRED data
fetch-data:
	python scripts/fetch_data.py

# Train models on historical data
train:
	python scripts/train_model.py

# Run a single nowcast and print results
nowcast:
	python scripts/run_nowcast.py

# Launch Streamlit dashboard
dashboard:
	streamlit run src/dashboard/app.py

# Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name "htmlcov" -exec rm -rf {} +
	find . -name ".coverage" -delete
