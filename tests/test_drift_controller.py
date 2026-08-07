from src.detection.model_loader import EdgePrediction
from src.drift.controller import DriftController
from src.drift.monitor import MultiFeatureDriftMonitor
from src.common.config import load_yaml_config


def build_controller(
    auto_approve: bool = False,
    approval_ttl_messages: int = 300,
    allow_approved_prediction_override: bool = False,
) -> DriftController:
    monitor = MultiFeatureDriftMonitor.from_config(
        [
            {
                "name": "voltage",
                "delta": 0.0,
                "threshold": 1.0,
                "minimum_instances": 2,
            }
        ]
    )
    return DriftController(
        monitor=monitor,
        adaptive_features=("voltage",),
        calibration_samples=2,
        trusted_confidence=0.9,
        maximum_trusted_anomaly=0.8,
        minimum_history=0,
        auto_approve=auto_approve,
        approval_ttl_messages=approval_ttl_messages,
        allow_approved_prediction_override=allow_approved_prediction_override,
        adapter_config={
            "window_size": 2,
            "minimum_samples": 2,
            "blend_rate": 1.0,
            "maximum_update_step": 1.0,
        },
    )


def features(voltage: float) -> dict[str, float | int]:
    return {
        "voltage": voltage,
        "history_count": 20,
        "device_topic_match": 1,
        "client_changed": 0,
        "topic_changed": 0,
        "unexpected_client_topic": 0,
        "is_duplicate_sequence": 0,
        "is_out_of_order": 0,
    }


def normal_prediction() -> EdgePrediction:
    return EdgePrediction("none", "none", 0.99, 0.2)


def attack_prediction() -> EdgePrediction:
    return EdgePrediction("random", "random", 0.99, 0.2)


def test_controller_calibrates_but_does_not_adapt_without_approval() -> None:
    controller = build_controller()
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.update("meter_01", features(230.0), normal_prediction())
    assert controller.reference_mean("meter_01", "voltage") == 230.0
    controller.update("meter_01", features(235.0), normal_prediction())
    result = controller.update("meter_01", features(235.0), normal_prediction())
    assert not result.adaptation_updated
    assert controller.reference_mean("meter_01", "voltage") == 230.0


def test_controller_updates_after_explicit_approval() -> None:
    controller = build_controller()
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.approve_drift("meter_01", "voltage")
    controller.update("meter_01", features(235.0), normal_prediction())
    result = controller.update("meter_01", features(235.0), normal_prediction())
    assert result.adaptation_updated
    assert controller.reference_mean("meter_01", "voltage") == 231.0
    assert result.reference_values == ("voltage=231.000000",)


def test_controller_rejects_protocol_anomaly() -> None:
    controller = build_controller()
    unsafe = features(230.0)
    unsafe["topic_changed"] = 1
    for _ in range(3):
        result = controller.update("meter_01", unsafe, normal_prediction())
    assert not result.adaptation_allowed
    assert controller.reference_mean("meter_01", "voltage") is None


def test_approved_drift_can_override_known_prediction_in_controlled_trial() -> None:
    controller = build_controller(
        allow_approved_prediction_override=True,
    )
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.approve_drift("meter_01", "voltage")
    first = controller.update("meter_01", features(235.0), attack_prediction())
    second = controller.update("meter_01", features(235.0), attack_prediction())
    assert first.adaptation_allowed
    assert second.adaptation_updated
    assert second.approved_features == ("voltage",)


def test_approved_prediction_override_is_disabled_by_default() -> None:
    controller = build_controller()
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.approve_drift("meter_01", "voltage")
    result = controller.update("meter_01", features(235.0), attack_prediction())
    assert not result.adaptation_allowed


def test_approval_expires_after_configured_number_of_messages() -> None:
    controller = build_controller(
        approval_ttl_messages=2,
        allow_approved_prediction_override=True,
    )
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.update("meter_01", features(230.0), normal_prediction())
    controller.approve_drift("meter_01", "voltage")
    first = controller.update("meter_01", features(235.0), attack_prediction())
    second = controller.update("meter_01", features(235.0), attack_prediction())
    expired = controller.update("meter_01", features(235.0), attack_prediction())
    assert first.approved_features == ("voltage",)
    assert second.approved_features == ("voltage",)
    assert expired.approved_features == ()
    assert not expired.adaptation_allowed


def test_edge_drift_config_is_loadable() -> None:
    config = load_yaml_config("config/edge.yaml")
    assert len(config["drift"]["features"]) == 2
    assert config["drift"]["adaptation"]["auto_approve"] is False
    experiment = load_yaml_config("config/edge_drift_experiment.yaml")
    assert experiment["drift"]["enabled"] is True
    assert experiment["drift"]["adaptation"]["auto_approve"] is True
    assert (
        experiment["drift"]["adaptation"][
            "allow_approved_prediction_override"
        ]
        is True
    )
