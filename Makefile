# Construction Labor Intel — common dev tasks.
#
# Targets are POSIX-portable and assume Python 3.11+ on PATH. Each one is a
# one-liner version of the README; the Makefile exists so the user doesn't
# have to remember the incantations.

.PHONY: help install install-dev install-dashboard install-training \
        test test-cov lint format \
        run dashboard demo \
        docker-up docker-down docker-logs \
        clean

PYTHON ?= python
PIP ?= pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------- Install ----------

install: ## Install backend runtime deps
	$(PIP) install -r requirements.txt

install-dev: install ## Backend deps + test/lint tools
	$(PIP) install pytest pytest-asyncio httpx ruff

install-dashboard: ## Streamlit dashboard deps
	$(PIP) install -r dashboard/requirements.txt

install-training: ## Heavy deps for retraining (torch, etc.)
	$(PIP) install -r requirements-training.txt

# ---------- Quality ----------

test: ## Run the test suite
	$(PYTHON) -m pytest -v

test-cov: ## Run tests with coverage report
	$(PIP) install -q coverage && \
	$(PYTHON) -m coverage run -m pytest && \
	$(PYTHON) -m coverage report -m

lint: ## Lint with ruff
	$(PYTHON) -m ruff check app/ tests/ scripts/ simulation/

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format app/ tests/ scripts/ simulation/

# ---------- Run ----------

run: ## Run backend with auto-reload (set LABOR_INTEL_STREAM_SOURCES first)
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dashboard: ## Run the Streamlit dashboard
	cd dashboard && streamlit run app.py

demo: ## Run end-to-end demo on a video file: make demo VIDEO=path.mp4 CAM=cam_01
	@test -n "$(VIDEO)" || (echo "Usage: make demo VIDEO=path.mp4 CAM=cam_01" && exit 1)
	@test -n "$(CAM)" || (echo "Usage: make demo VIDEO=path.mp4 CAM=cam_01" && exit 1)
	$(PYTHON) -m scripts.demo_video --video "$(VIDEO)" --camera-id "$(CAM)" \
	  --output ./demo_output

# ---------- Docker ----------

docker-up: ## docker compose up --build -d
	docker compose up --build -d

docker-down: ## docker compose down
	docker compose down

docker-logs: ## Tail logs from the backend
	docker compose logs -f backend

# ---------- Hygiene ----------

clean: ## Remove caches and demo output
	rm -rf .pytest_cache .ruff_cache .coverage __pycache__ \
	       app/__pycache__ tests/__pycache__ scripts/__pycache__ simulation/__pycache__ \
	       demo_output/
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
