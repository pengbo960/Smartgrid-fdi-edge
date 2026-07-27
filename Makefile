PYTHON := python

SCENARIO_DIR := config/scenarios/dataset
SMOKE_SCENARIO_DIR := config/scenarios/smoke

RAW_DIR := data/raw/training_runs
SMOKE_RAW_DIR := data/raw/smoke_runs

PROCESSED_DATASET := data/processed/multiview_dataset.csv

.PHONY: help test scenarios smoke-scenarios collect smoke-collect features clean-smoke validate ablation

help:
	@echo "Available commands:"
	@echo "  make test"
	@echo "  make scenarios"
	@echo "  make smoke-scenarios"
	@echo "  make collect"
	@echo "  make smoke-collect"
	@echo "  make features"
	@echo "  make clean-smoke"
	@echo "  make validate"
	@echo "  make ablation"

test:
	$(PYTHON) -m pytest -v

scenarios:
	$(PYTHON) scripts/generate_scenarios.py \
		--runs 3 \
		--duration 300 \
		--interval 0.5 \
		--output-dir $(SCENARIO_DIR)

smoke-scenarios:
	$(PYTHON) scripts/generate_scenarios.py \
		--runs 1 \
		--duration 8 \
		--interval 0.2 \
		--output-dir $(SMOKE_SCENARIO_DIR)

collect:
	./scripts/run_all_scenarios.sh \
		$(SCENARIO_DIR) \
		$(RAW_DIR)

smoke-collect:
	./scripts/run_all_scenarios.sh \
		$(SMOKE_SCENARIO_DIR) \
		$(SMOKE_RAW_DIR)

features:
	$(PYTHON) scripts/build_dataset.py \
		--input-dir $(RAW_DIR) \
		--output $(PROCESSED_DATASET)

clean-smoke:
	rm -rf $(SMOKE_SCENARIO_DIR)
	rm -rf $(SMOKE_RAW_DIR)

validate:
	$(PYTHON) scripts/validate_dataset.py \
		--input-dir $(RAW_DIR)

ablation:
	$(PYTHON) scripts/run_ablation.py \
		--config config/ablation.yaml