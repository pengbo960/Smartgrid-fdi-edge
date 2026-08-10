PYTHON := python

SCENARIO_DIR := config/scenarios/dataset
SMOKE_SCENARIO_DIR := config/scenarios/smoke

RAW_DIR := data/raw/training_runs
SMOKE_RAW_DIR := data/raw/smoke_runs

PROCESSED_DATASET := data/processed/multiview_dataset.csv

.PHONY: help test scenarios smoke-scenarios collect smoke-collect features clean-smoke validate ablation open-set edge-detector edge-benchmark edge-benchmark-repeated compare-models repeated-experiments drift drift-repeated final-summary experiments

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
	@echo "  make open-set"
	@echo "  make edge-detector"
	@echo "  make edge-benchmark INPUT=data/raw/training_runs/normal_run_01.csv"
	@echo "  make edge-benchmark-repeated"
	@echo "  make compare-models"
	@echo "  make repeated-experiments"
	@echo "  make drift"
	@echo "  make drift-repeated"
	@echo "  make final-summary"
	@echo "  make experiments"

test:
	$(PYTHON) -m pytest -v

scenarios:
	$(PYTHON) scripts/generate_scenarios.py \
		--runs 5 \
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

open-set:
	$(PYTHON) scripts/train_open_set.py \
		--config config/open_set.yaml

edge-detector:
	$(PYTHON) scripts/run_edge_detector.py \
		--config config/edge.yaml

edge-benchmark:
	$(PYTHON) scripts/benchmark_edge.py \
		--config config/edge.yaml \
		--input $(INPUT)

edge-benchmark-repeated:
	$(PYTHON) scripts/run_repeated_edge_benchmark.py \
		--config config/repeated_edge_benchmark.yaml

compare-models:
	$(PYTHON) scripts/compare_models.py \
		--config config/model_comparison.yaml

repeated-experiments:
	$(PYTHON) scripts/run_repeated_experiments.py \
		--config config/repeated_experiments.yaml

drift:
	$(PYTHON) scripts/evaluate_drift.py \
		--config config/drift.yaml

drift-repeated:
	$(PYTHON) scripts/evaluate_drift_repeated.py \
		--config config/drift.yaml

final-summary:
	$(PYTHON) scripts/summarize_results.py

experiments:
	$(PYTHON) scripts/run_ablation.py --config config/ablation.yaml
	$(PYTHON) scripts/train_open_set.py --config config/open_set.yaml
	$(PYTHON) scripts/compare_models.py --config config/model_comparison.yaml
	$(PYTHON) scripts/evaluate_drift.py --config config/drift.yaml
	$(PYTHON) scripts/evaluate_drift_repeated.py --config config/drift.yaml
	$(PYTHON) scripts/summarize_results.py
