#!/usr/bin/env bash

set -euo pipefail

SCENARIO_DIR="${1:-config/scenarios/dataset}"
OUTPUT_DIR="${2:-data/raw/training_runs}"

mkdir -p "$OUTPUT_DIR"

for scenario in "$SCENARIO_DIR"/*.yaml
do
    echo
    echo "========================================"
    echo "Running: $scenario"
    echo "========================================"

    python scripts/run_collection.py \
        --scenario "$scenario" \
        --output-dir "$OUTPUT_DIR"
done

echo
echo "All scenarios completed."