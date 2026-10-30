.PHONY: install lint format test docker-build healthcheck demo api dashboard up down

install:      ; pip install -r requirements.txt && pre-commit install
lint:         ; ruff check . && ruff format --check . && mypy src
format:       ; ruff check --fix . && ruff format .
test:         ; pytest -m "not azure and not slow"
docker-build: ; docker compose build
healthcheck:  ; medwatch healthcheck

# fluxo completo no terminal: cenário → fusão → alertas, conferido contra a sequência documentada
demo:         ; medwatch simulate --scenario demo
# tempo real com API + dashboard (dois terminais) e depois: medwatch simulate --target api
api:          ; medwatch api
dashboard:    ; medwatch dashboard
# o mesmo com Docker: API em :8000, dashboard em :8501 e o simulador da demo
up:           ; docker compose --profile demo up --build
down:         ; docker compose --profile demo down
