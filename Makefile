.DEFAULT_GOAL := help
DATABASE_URL ?= postgresql://rag:rag@localhost:5433/rag
export DATABASE_URL

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Start Postgres and wait for it
	docker compose up -d --wait

down: ## Stop Postgres (keeps data)
	docker compose down

clean: ## Stop Postgres and delete the volume
	docker compose down -v

install: ## Install the package with embedding + dev extras
	pip install -e ".[embed,dev]"

init: ## Create the schema
	python -m ragkit init

index: ## Index ./corpus
	python -m ragkit index corpus

search: ## Search: make search Q="your question"
	python -m ragkit search "$(Q)"

stats: ## Index statistics
	python -m ragkit stats

test: ## Unit tests (no database, no model download)
	pytest -q

test-all: ## Every test, including Postgres-backed ones
	pytest -q -m "not model"

lint: ## Lint and format check
	ruff check . && ruff format --check .

fmt: ## Format
	ruff format .

demo: up install init index ## Full path from nothing to a working index
	@echo
	@python -m ragkit search "how long do I have to return an international order?"

.PHONY: help up down clean install init index search stats test test-all lint fmt demo

eval: ## Measure retrieval against the golden set
	python -m ragkit eval --detail

eval-compare: ## Compare dense / lexical / hybrid / +rerank
	python -m ragkit eval --compare

eval-check: ## Fail if retrieval regressed below evals/baseline.json
	python -m ragkit eval --check-floor

eval-baseline: ## Re-record the regression floor from current metrics
	python -m ragkit eval --save-baseline

.PHONY: eval eval-compare eval-check eval-baseline

serve: ## Run the HTTP API on :8080
	uvicorn ragkit.service:app --port 8080 --reload

mcp: ## Run the MCP server on stdio (for manual inspection)
	python -m ragkit.mcp_server

.PHONY: serve mcp
