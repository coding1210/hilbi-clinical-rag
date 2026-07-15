# Convenience targets. Everything runs offline (mock LLM) unless you export
# ANTHROPIC_API_KEY, in which case `make eval` / `make ask` use Claude.

PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup index eval ask test clean

setup:
	uv venv --python $(PY) $(VENV) || $(PY) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/python -m spacy download en_core_web_lg

index:
	$(BIN)/python -m scripts.build_index

eval:
	$(BIN)/python -m scripts.run_eval

# Usage: make ask Q="65yo M John Smith MRN 12345 with chest pain, what workup?"
ask:
	$(BIN)/python -m scripts.ask "$(Q)"

test:
	$(BIN)/python -m pytest -q

clean:
	rm -rf $(VENV) data/index results/eval_results.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
