import pytest

from src.drift.guarded_adaptation import ReferenceAdapter
from src.drift.monitor import MultiFeatureDriftMonitor
from src.drift.page_hinkley import PageHinkley


def test_page_hinkley_detects_increase() -> None:
    detector = PageHinkley(
        delta=0.01, threshold=2.0, minimum_instances=20
    )
    events = []
    for value in [0.0] * 50 + [1.0] * 50:
        events.append(detector.update(value))
    detected = [event for event in events if event.drift_detected]
    assert detected
    assert detected[0].direction == "increase"


def test_page_hinkley_stable_stream_has_no_drift() -> None:
    detector = PageHinkley(
        delta=0.01, threshold=2.0, minimum_instances=20
    )
    assert not any(
        detector.update(0.0).drift_detected for _ in range(100)
    )


def test_page_hinkley_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        PageHinkley().update(float("nan"))


def test_guarded_adapter_requires_confirmation_and_trust() -> None:
    adapter = ReferenceAdapter(
        230.0, window_size=4, minimum_samples=2, guarded=True
    )
    blocked = adapter.update(235.0, trusted_sample=True, drift_confirmed=False)
    untrusted = adapter.update(235.0, trusted_sample=False, drift_confirmed=True)
    assert not blocked.accepted
    assert not untrusted.accepted
    assert adapter.reference_mean == 230.0


def test_guarded_adapter_updates_bounded_reference() -> None:
    adapter = ReferenceAdapter(
        230.0,
        window_size=4,
        minimum_samples=2,
        blend_rate=1.0,
        maximum_update_step=1.0,
        guarded=True,
    )
    adapter.update(235.0, trusted_sample=True, drift_confirmed=True)
    result = adapter.update(235.0, trusted_sample=True, drift_confirmed=True)
    assert result.updated
    assert result.reference_mean == 231.0


def test_unguarded_adapter_accepts_unconfirmed_samples() -> None:
    adapter = ReferenceAdapter(
        230.0,
        window_size=2,
        minimum_samples=2,
        blend_rate=1.0,
        maximum_update_step=10.0,
        guarded=False,
    )
    adapter.update(240.0, trusted_sample=False, drift_confirmed=False)
    result = adapter.update(240.0, trusted_sample=False, drift_confirmed=False)
    assert result.updated
    assert result.reference_mean == 240.0


def test_multi_feature_monitor_is_per_device() -> None:
    monitor = MultiFeatureDriftMonitor.from_config(
        [
            {
                "name": "voltage",
                "delta": 0.01,
                "threshold": 2.0,
                "minimum_instances": 20,
            }
        ]
    )
    for _ in range(50):
        assert monitor.update("meter_01", {"voltage": 230.0}) == []
        assert monitor.update("meter_02", {"voltage": 230.0}) == []
    events = []
    for _ in range(20):
        events.extend(monitor.update("meter_01", {"voltage": 235.0}))
        monitor.update("meter_02", {"voltage": 230.0})
    assert events
    assert all(event["device_id"] == "meter_01" for event in events)
