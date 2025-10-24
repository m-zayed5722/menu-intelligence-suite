.PHONY: help dev test fmt lint bench clean

help:
	@echo "Menu Intelligence Suite - Makefile Commands"
	@echo "============================================"
	@echo "make dev        - Start development environment"
	@echo "make test       - Run tests with coverage"
	@echo "make fmt        - Format code with black"
	@echo "make lint       - Lint code with ruff"
	@echo "make bench      - Run benchmark and generate report"
	@echo "make clean      - Clean up containers and volumes"
	@echo "make seed       - Generate and load seed data"

dev:
	@echo "Starting development environment..."
	@if not exist .env copy .env.example .env
	@docker-compose up --build -d
	@echo "Waiting for services to be healthy..."
	@timeout /t 10 /nobreak >nul
	@echo "Running migrations..."
	@docker-compose exec -T db psql -U postgres -d mis -f /docker-entrypoint-initdb.d/001_init.sql || echo "Migration already applied"
	@docker-compose exec -T db psql -U postgres -d mis -f /docker-entrypoint-initdb.d/002_indexes.sql || echo "Indexes already created"
	@echo ""
	@echo "✓ Services started!"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Streamlit:  http://localhost:8501"
	@echo ""
	@echo "Run 'make seed' to load sample data"

seed:
	@echo "Generating seed data..."
	@docker-compose exec api python src/data/generate_data.py
	@echo "✓ Seed data generated and loaded"

test:
	@echo "Running tests..."
	@docker-compose exec api pytest tests/ -v --cov=src --cov-report=term-missing

fmt:
	@echo "Formatting code..."
	@black src/ tests/
	@echo "✓ Code formatted"

lint:
	@echo "Linting code..."
	@ruff check src/ tests/
	@echo "✓ Linting complete"

bench:
	@echo "Running benchmarks..."
	@if not exist reports mkdir reports
	@docker-compose exec api python -m pytest tests/test_search.py::test_benchmark -v > reports\benchmark.txt
	@echo "✓ Benchmark report saved to reports\benchmark.txt"

clean:
	@echo "Cleaning up..."
	@docker-compose down -v
	@echo "✓ Cleanup complete"

logs:
	@docker-compose logs -f api

db-shell:
	@docker-compose exec db psql -U postgres -d mis
