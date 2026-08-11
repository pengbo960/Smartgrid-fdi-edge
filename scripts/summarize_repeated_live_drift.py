from __future__ import annotations

import argparse
from pathlib import Path

from src.drift.repeated_live_evaluation import evaluate_repeated_live_drift
from src.evaluation.repeated_experiments import save_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate independent repeated live MQTT drift trials."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-directory", type=Path,
        default=Path("results/drift/repeated_live"),
    )
    args = parser.parse_args()
    runs, run_summary, phases, phase_summary = evaluate_repeated_live_drift(
        args.inputs
    )
    output = args.output_directory
    save_table(runs, output / "live_drift_runs.csv")
    save_table(run_summary, output / "live_drift_summary.csv")
    save_table(phases, output / "live_drift_phase_runs.csv")
    save_table(phase_summary, output / "live_drift_phase_summary.csv")
    print("\nOVERALL REPEATED LIVE DRIFT")
    print(run_summary.to_string(index=False))
    print("\nPHASE-WISE REPEATED LIVE DRIFT")
    print(phase_summary.to_string(index=False))
    print(f"\nResults saved under: {output}")


if __name__ == "__main__":
    main()
