.DEFAULT_GOAL := help
SHELL := /bin/bash

# Every recipe runs through uv so the lockfile is always the source of truth.
UV := uv

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Local development ────────────────────────────────────────────────────
.PHONY: install
install: ## Create the virtualenv and install all dependencies
	$(UV) sync --all-extras

.PHONY: api
api: ## Run the API with autoreload (needs postgres + redis)
	$(UV) run uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000

.PHONY: ui
ui: ## Run the Streamlit client
	$(UV) run streamlit run ui/streamlit_app.py --server.port 8501

.PHONY: mcp
mcp: ## Run the MCP server
	$(UV) run python -m mcp_server.server

# ── Quality ──────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run unit + security suites (no external services required)
	$(UV) run pytest tests/unit tests/security -q

.PHONY: test-all
test-all: ## Run every suite, including tests that need live services
	$(UV) run pytest -q

.PHONY: lint
lint: ## Lint and check formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: fmt
fmt: ## Autoformat and apply safe lint fixes
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: types
types: ## Static type check
	$(UV) run mypy app

.PHONY: check
check: lint types test ## Everything CI would run

# ── Data ─────────────────────────────────────────────────────────────────
.PHONY: corpus
corpus: ## Generate the synthetic document corpus
	$(UV) run python scripts/generate_corpus.py

.PHONY: mcp-data
mcp-data: corpus ## Generate the MCP server's enterprise data from the corpus
	$(UV) run python scripts/generate_mcp_data.py

.PHONY: seed
seed: corpus mcp-data ## Generate all data and index it into Pinecone + local BM25
	$(UV) run python scripts/index_corpus.py

# ── Compose ──────────────────────────────────────────────────────────────
.PHONY: up
up: ## Build and start every service
	docker compose up --build -d
	@echo
	@echo "  UI    http://localhost:8501"
	@echo "  API   http://localhost:8000/docs"
	@echo "  Deps  http://localhost:8000/health/deps"

.PHONY: down
down: ## Stop every service
	docker compose down

.PHONY: logs
logs: ## Tail service logs
	docker compose logs -f api ui mcp

.PHONY: clean
clean: ## Remove containers, volumes, and local caches
	docker compose down -v
	rm -rf .pytest_cache .ruff_cache .mypy_cache data/bm25_index
