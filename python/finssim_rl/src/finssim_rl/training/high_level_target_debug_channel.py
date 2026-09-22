from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from mlagents_envs.side_channel.side_channel import IncomingMessage, OutgoingMessage, SideChannel


HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ID = uuid.UUID("6f5d8f67-3d91-4f69-9acb-18d90e2f4a31")
HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ATTR = "finsim_high_level_target_debug_channel"


class HighLevelTargetDebugChannel(SideChannel):
    """Python -> Unity debug channel for 1Chase1 hierarchy high-level subgoals."""

    def __init__(self) -> None:
        super().__init__(HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:  # noqa: D401 - ML-Agents hook
        del msg

    def send_high_level_target(
        self,
        *,
        step_index: int,
        local_target_delta: Sequence[float],
        target_world: Sequence[float],
        current_world: Sequence[float],
    ) -> None:
        outgoing = OutgoingMessage()
        outgoing.write_int32(int(step_index))
        outgoing.write_float32_list([float(v) for v in local_target_delta])
        outgoing.write_float32_list([float(v) for v in target_world])
        outgoing.write_float32_list([float(v) for v in current_world])
        self.queue_message_to_send(outgoing)


def attach_high_level_target_debug_channel(env: Any, channel: HighLevelTargetDebugChannel) -> None:
    setattr(env, HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ATTR, channel)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:
        setattr(unwrapped, HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ATTR, channel)


def resolve_high_level_target_debug_channel(env: Any) -> Optional[HighLevelTargetDebugChannel]:
    if env is None:
        return None

    seen: set[int] = set()
    stack: list[Any] = [env]
    while stack:
        current = stack.pop()
        if current is None:
            continue

        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        channel = getattr(current, HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ATTR, None)
        if isinstance(channel, HighLevelTargetDebugChannel):
            return channel

        for attr_name in ("env", "venv", "unwrapped"):
            child = getattr(current, attr_name, None)
            if child is not None and child is not current:
                stack.append(child)

        envs = getattr(current, "envs", None)
        if isinstance(envs, (list, tuple)):
            stack.extend(envs)

    return None


__all__ = [
    "HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ATTR",
    "HIGH_LEVEL_TARGET_DEBUG_CHANNEL_ID",
    "HighLevelTargetDebugChannel",
    "attach_high_level_target_debug_channel",
    "resolve_high_level_target_debug_channel",
]
