import pandas as pd
import pytest

from src.evaluation.deployment_parity import (
    compare_prediction_manifests,
    prediction_fingerprint,
)


def manifest() -> pd.DataFrame:
    return pd.DataFrame({
        "input_row_index": [10, 11],
        "device_id": ["meter_01", "meter_02"],
        "sequence_number": [3, 3],
        "true_attack_type": ["none", "gradual"],
        "known_prediction": ["none", "random"],
        "decision": ["none", "unknown"],
        "confidence": [0.99, 0.51],
        "anomaly_score": [0.1, 0.8],
    })


def test_identical_prediction_manifests_pass() -> None:
    reference = manifest()
    report, mismatches = compare_prediction_manifests(reference, reference.copy())
    assert report["parity_passed"] is True
    assert report["overall_match_rate"] == 1.0
    assert report["decision_fingerprints_match"] is True
    assert mismatches.empty


def test_small_score_difference_respects_tolerance() -> None:
    reference = manifest()
    candidate = manifest()
    candidate.loc[1, "confidence"] += 1e-13
    report, _ = compare_prediction_manifests(reference, candidate)
    assert report["parity_passed"] is True
    assert report["maximum_confidence_difference"] == pytest.approx(1e-13)


def test_decision_difference_fails_parity() -> None:
    reference = manifest()
    candidate = manifest()
    candidate.loc[1, "decision"] = "known_attack"
    report, mismatches = compare_prediction_manifests(reference, candidate)
    assert report["parity_passed"] is False
    assert report["decision_match_rate"] == 0.5
    assert len(mismatches) == 1
    assert prediction_fingerprint(reference) != prediction_fingerprint(candidate)


def test_duplicate_message_index_is_rejected() -> None:
    candidate = manifest()
    candidate.loc[1, "input_row_index"] = 10
    with pytest.raises(ValueError, match="duplicate"):
        compare_prediction_manifests(manifest(), candidate)
