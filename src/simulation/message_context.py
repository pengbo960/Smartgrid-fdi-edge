from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageContext:
    """
    MQTT routing metadata that may be manipulated independently of data.
    """

    topic: str
    client_id: str

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError(
                "topic must not be empty"
            )

        if not self.client_id:
            raise ValueError(
                "client_id must not be empty"
            )


def apply_topic_spoof(
    context: MessageContext,
    spoofed_device_id: str,
) -> MessageContext:
    """
    Replace the device level in an MQTT measurement topic.

    The payload device_id remains unchanged so the gateway can observe
    the inconsistency between the claimed identity and routing context.
    """
    if not spoofed_device_id:
        raise ValueError(
            "spoofed_device_id must not be empty"
        )

    topic_levels = context.topic.split("/")

    if (
        len(topic_levels) < 2
        or topic_levels[-1] != "measurement"
    ):
        raise ValueError(
            "Topic must end with a device/measurement suffix"
        )

    topic_levels[-2] = spoofed_device_id

    return MessageContext(
        topic="/".join(topic_levels),
        client_id=context.client_id,
    )
