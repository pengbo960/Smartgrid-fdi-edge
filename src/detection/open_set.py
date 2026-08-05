from __future__ import annotations

import numpy as np


def quantile_threshold(
    values: np.ndarray | list[float],
    quantile: float,
) -> float:
    """
    Select a threshold from validation-only scores.
    """
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "Threshold values must be a non-empty "
            "one-dimensional array"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Threshold values must be finite"
        )

    if not 0 <= quantile <= 1:
        raise ValueError(
            "quantile must be between 0 and 1"
        )

    return float(
        np.quantile(
            array,
            quantile,
        )
    )


def apply_open_set_decision(
    predicted_labels: np.ndarray | list[str],
    confidence_scores: np.ndarray | list[float],
    anomaly_scores: np.ndarray | list[float],
    confidence_threshold: float,
    anomaly_threshold: float,
    normal_label: str = "none",
    unknown_label: str = "unknown",
) -> np.ndarray:
    """
    Reject low-confidence predictions and anomalous normal predictions.

    The anomaly detector is fitted only on normal training samples. Its
    score is therefore used only when the supervised classifier predicts
    normal; applying it to known attacks would incorrectly reject attacks
    that are expected to differ from normal behaviour.
    """
    labels = np.asarray(
        predicted_labels,
        dtype=object,
    )
    confidence = np.asarray(
        confidence_scores,
        dtype=float,
    )
    anomaly = np.asarray(
        anomaly_scores,
        dtype=float,
    )

    if labels.ndim != 1:
        raise ValueError(
            "predicted_labels must be one-dimensional"
        )

    if confidence.ndim != 1 or anomaly.ndim != 1:
        raise ValueError(
            "Score arrays must be one-dimensional"
        )

    if not (
        len(labels)
        == len(confidence)
        == len(anomaly)
    ):
        raise ValueError(
            "Labels and score arrays must have equal length"
        )

    if not (
        np.isfinite(confidence).all()
        and np.isfinite(anomaly).all()
    ):
        raise ValueError(
            "Decision scores must be finite"
        )

    if not 0 <= confidence_threshold <= 1:
        raise ValueError(
            "confidence_threshold must be between 0 and 1"
        )

    decisions = labels.astype(
        object,
        copy=True,
    )

    reject = (
        confidence < confidence_threshold
    ) | (
        (labels == normal_label)
        & (anomaly > anomaly_threshold)
    )

    decisions[reject] = unknown_label

    return decisions
