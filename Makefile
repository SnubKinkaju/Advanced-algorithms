PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
METHOD ?= trivial
INSTANCE ?= data/benchmarks/e/scpe1.txt
TIME_LIMIT ?= 10
SEED ?= 0
OUTPUT ?=
ELEMENTS ?= 20
SUBSETS ?= 40
DENSITY ?= 0.2
GENERATED ?= data/generated/example.txt
TEST_PATH ?= tests

.DEFAULT_GOAL := help
.PHONY: help setup test methods run generate bounds clean

help:
	@echo "make setup      Create the virtual environment and install dependencies"
	@echo "make test       Run tests (optional: TEST_PATH=tests/test_io.py)"
	@echo "make methods    List registered algorithms"
	@echo "make run        Run a method and check its solution"
	@echo "                METHOD=trivial INSTANCE=data/benchmarks/e/scpe1.txt TIME_LIMIT=10 SEED=0"
	@echo "                Optional: OUTPUT=results/scratch/run.json"
	@echo "make generate   Generate an instance: ELEMENTS=20 SUBSETS=40 DENSITY=0.2 SEED=0"
	@echo "                GENERATED=data/generated/example.txt (replaces the output file)"
	@echo "make bounds     Compute counting and LP lower bounds for INSTANCE"
	@echo "make clean      Remove Python and pytest caches"

setup: $(VENV)/.installed

$(VENV)/.installed: pyproject.toml
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install -e ".[dev]"
	touch "$@"

test: setup
	"$(VENV_PYTHON)" -m pytest -q "$(TEST_PATH)"

methods: setup
	@"$(VENV_PYTHON)" -m setcover methods

run: setup
	@"$(VENV_PYTHON)" -m setcover run --method "$(METHOD)" --instance "$(INSTANCE)" --time-limit "$(TIME_LIMIT)" --seed "$(SEED)" --output "$(OUTPUT)"

generate: setup
	@"$(VENV_PYTHON)" -m setcover generate --elements "$(ELEMENTS)" --subsets "$(SUBSETS)" --density "$(DENSITY)" --seed "$(SEED)" --output "$(GENERATED)"

bounds: setup
	@"$(VENV_PYTHON)" -m setcover bounds --instance "$(INSTANCE)"

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
