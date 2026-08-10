import numpy as np
import pytest

from src.detection.comparison_model_loader import ComparisonModelBundle


class FakeModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probability = np.clip(values[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))


def build_bundle() -> ComparisonModelBundle:
    return ComparisonModelBundle(
        model=FakeModel(), scaler=None, feature_columns=("feature",),
        threshold=0.6, model_name="fake", artifact_path="fake.joblib",
    )


def test_comparison_bundle_applies_threshold() -> None:
    assert build_bundle().predict({"feature": 0.7}).decision == "known_attack"
    assert build_bundle().predict({"feature": 0.4}).decision == "none"


def test_comparison_bundle_rejects_missing_or_non_finite_features() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_bundle().predict({})
    with pytest.raises(ValueError, match="finite"):
        build_bundle().predict({"feature": float("nan")})


def test_comparison_bundle_validates_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ComparisonModelBundle(
            FakeModel(), None, ("feature",), 1.1, "fake", "fake.joblib"
        )
