from __future__ import annotations

import argparse
import json

import pandas as pd

from src.evaluation.deployment_parity import (
    compare_prediction_manifests,
    save_parity_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MacBook and Raspberry Pi deployment predictions."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mismatches", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--relative-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    report, mismatches = compare_prediction_manifests(
        pd.read_csv(args.reference),
        pd.read_csv(args.candidate),
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    save_parity_report(report, mismatches, args.report, args.mismatches)
    print(json.dumps(report, indent=2))
    print(f"Mismatched rows: {len(mismatches)}")
    print(f"Saved report to: {args.report}")
    if not report["parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
