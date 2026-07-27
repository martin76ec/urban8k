PYTHON := uv run python
PYTEST := uv run pytest
CONFIG ?= configs/h200.yaml
RUN_ID ?=

.PHONY: setup lint format typecheck test smoke data train evaluate all

setup:
	uv sync --group dev

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

typecheck:
	uv run mypy src

test:
	$(PYTEST) tests/unit

smoke:
	$(PYTEST) -m smoke tests/smoke

data:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) $(if $(RUN_ID),--run-id $(RUN_ID))

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --run-id $(RUN_ID)

all: lint typecheck test smoke