from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = (
    "input_row_index",
    "device_id",
    "sequence_number",
    "true_attack_type",
)

CATEGORICAL_COLUMNS = (
    "known_prediction",
    "decision",
)

NUMERIC_COLUMNS = (
    "confidence",
    "anomaly_score",
)


def prediction_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered message identities and categorical deployment decisions."""
    columns = [*KEY_COLUMNS, *CATEGORICAL_COLUMNS]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction manifest is missing columns: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in frame[columns].itertuples(index=False, name=None):
        encoded = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def compare_prediction_manifests(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare categorical decisions exactly and floating scores tolerantly."""
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Parity tolerances must not be negative")
    required = set(KEY_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)
    for label, frame in (("reference", reference), ("candidate", candidate)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} prediction manifest is missing columns: {sorted(missing)}"
            )
        if frame["input_row_index"].duplicated().any():
            raise ValueError(f"{label} manifest contains duplicate input_row_index")

    merged = reference.merge(
        candidate,
        on="input_row_index",
        how="outer",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
        indicator=True,
    )
    identity_match = merged["_merge"].eq("both")
    for column in KEY_COLUMNS[1:]:
        identity_match &= (
            merged[f"{column}_reference"].astype(str)
            == merged[f"{column}_candidate"].astype(str)
        )
    for column in CATEGORICAL_COLUMNS:
        merged[f"{column}_match"] = (
            merged[f"{column}_reference"].astype(str)
            == merged[f"{column}_candidate"].astype(str)
        ) & merged["_merge"].eq("both")
    for column in NUMERIC_COLUMNS:
        reference_values = pd.to_numeric(
            merged[f"{column}_reference"], errors="coerce"
        ).to_numpy(dtype=float)
        candidate_values = pd.to_numeric(
            merged[f"{column}_candidate"], errors="coerce"
        ).to_numpy(dtype=float)
        merged[f"{column}_absolute_difference"] = np.abs(
            reference_values - candidate_values
        )
        merged[f"{column}_match"] = np.isclose(
            reference_values,
            candidate_values,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            equal_nan=False,
        ) & merged["_merge"].eq("both").to_numpy()

    all_match = identity_match.copy()
    for column in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        all_match &= merged[f"{column}_match"]
    merged["all_match"] = all_match
    compared = len(merged)
    categorical_match = np.logical_and.reduce([
        merged[f"{column}_match"].to_numpy(dtype=bool)
        for column in CATEGORICAL_COLUMNS
    ])
    numeric_match = np.logical_and.reduce([
        merged[f"{column}_match"].to_numpy(dtype=bool)
        for column in NUMERIC_COLUMNS
    ])
    report: dict[str, Any] = {
        "reference_messages": len(reference),
        "candidate_messages": len(candidate),
        "compared_rows": compared,
        "identity_matches": int(identity_match.sum()),
        "identity_match_rate": float(identity_match.mean()) if compared else 0.0,
        "known_prediction_matches": int(
            merged["known_prediction_match"].sum()
        ),
        "known_prediction_match_rate": float(
            merged["known_prediction_match"].mean()
        ) if compared else 0.0,
        "decision_matches": int(merged["decision_match"].sum()),
        "decision_match_rate": float(merged["decision_match"].mean())
        if compared else 0.0,
        "categorical_match_rate": float(categorical_match.mean())
        if compared else 0.0,
        "numeric_match_rate": float(numeric_match.mean()) if compared else 0.0,
        "overall_match_rate": float(all_match.mean()) if compared else 0.0,
        "maximum_confidence_difference": float(
            merged["confidence_absolute_difference"].max()
        ) if compared else None,
        "maximum_anomaly_score_difference": float(
            merged["anomaly_score_absolute_difference"].max()
        ) if compared else None,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "reference_decision_fingerprint": prediction_fingerprint(reference),
        "candidate_decision_fingerprint": prediction_fingerprint(candidate),
        "decision_fingerprints_match": (
            prediction_fingerprint(reference) == prediction_fingerprint(candidate)
        ),
        "parity_passed": bool(all_match.all()) and len(reference) == len(candidate),
    }
    mismatches = merged.loc[~merged["all_match"]].copy()
    return report, mismatches


def save_parity_report(
    report: dict[str, Any],
    mismatches: pd.DataFrame,
    report_path: str | Path,
    mismatches_path: str | Path,
) -> None:
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    mismatch_destination = Path(mismatches_path)
    mismatch_destination.parent.mkdir(parents=True, exist_ok=True)
    mismatches.to_csv(mismatch_destination, index=False)
