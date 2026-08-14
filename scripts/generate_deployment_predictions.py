from __future__ import annotations

import argparse
import platform
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.detection.edge_detector import EdgeDetector
from src.detection.model_loader import OpenSetModelBundle
from src.evaluation.edge_warmup import partition_device_warmup
from src.evaluation.deployment_parity import prediction_fingerprint
from src.features.data_loader import load_raw_dataset
from src.features.feature_pipeline import StreamingFeaturePipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic predictions for deployment parity."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-per-device", type=int, default=34)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    artifacts = config["artifacts"]
    features = config["features"]
    detector = EdgeDetector(
        model=OpenSetModelBundle.load(
            artifacts["classifier"],
            artifacts["scaler"],
            artifacts["anomaly_detector"],
            artifacts["metadata"],
        ),
        feature_pipeline=StreamingFeaturePipeline(
            window_size=int(features["window_size"]),
            minimum_history=int(features["minimum_history"]),
            power_factor=float(features["power_factor"]),
            repeated_value_field=str(features["repeated_value_field"]),
            value_tolerance=float(features["value_tolerance"]),
        ),
    )
    source_rows = load_raw_dataset(args.input).to_dict(orient="records")
    indexed_rows = [
        {**row, "_input_row_index": index}
        for index, row in enumerate(source_rows)
    ]
    warmup_rows, measured_rows, _ = partition_device_warmup(
        indexed_rows, args.warmup_per_device
    )
    for row in warmup_rows:
        detector.process(row)
    records: list[dict[str, object]] = []
    for row in measured_rows:
        prediction = detector.process(row)
        records.append({
            "input_row_index": int(row["_input_row_index"]),
            "device_id": str(row["device_id"]),
            "sequence_number": int(row["sequence_number"]),
            "true_attack_type": str(row.get("attack_type", "")),
            "known_prediction": str(prediction["known_prediction"]),
            "decision": str(prediction["decision"]),
            "confidence": float(prediction["confidence"]),
            "anomaly_score": float(prediction["anomaly_score"]),
        })
    manifest = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False, float_format="%.17g")
    print(f"Platform: {config['platform_label']}")
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Messages: {len(manifest)}")
    print(f"Decision fingerprint: {prediction_fingerprint(manifest)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
