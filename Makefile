.PHONY: install lint format test docker-build healthcheck

install:      ; pip install -r requirements.txt && pre-commit install
lint:         ; ruff check . && ruff format --check . && mypy src
format:       ; ruff check --fix . && ruff format .
test:         ; pytest -m "not azure and not slow"
docker-build: ; docker compose build
healthcheck:  ; medwatch healthcheck
