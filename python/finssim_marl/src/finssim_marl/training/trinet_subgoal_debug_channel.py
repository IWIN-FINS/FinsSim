"""Python-to-Unity debug transport for TriNetCapture baseline subgoals."""

from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from mlagents_envs.side_channel.side_channel import IncomingMessage, OutgoingMessage, SideChannel


TRINET_SUBGOAL_DEBUG_CHANNEL_ID = uuid.UUID("d90d4182-49cb-4e1f-8319-43423249c46d")
TRINET_SUBGOAL_DEBUG_CHANNEL_ATTR = "finssim_trinet_subgoal_debug_channel"


class TriNetSubgoalDebugChannel(SideChannel):
    """One-way debug channel carrying one 4D PID subgoal per TriNet ROV."""

    def __init__(self) -> None:
        super().__init__(TRINET_SUBGOAL_DEBUG_CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:  # noqa: D401 - ML-Agents hook
        del msg

    def send_subgoals(
        self,
        *,
        step_index: int,
        area_index: int,
        shared_translation: Sequence[Sequence[float]],
        formation_correction: Sequence[Sequence[float]],
        body_subgoal: Sequence[Sequence[float]],
        yaw_error_deg: Sequence[float],
        phase: Sequence[int],
    ) -> None:
        if not (
            len(shared_translation) == len(formation_correction) == len(body_subgoal)
            == len(yaw_error_deg) == len(phase) == 3
        ):
            raise ValueError("TriNet debug payload requires exactly three ROV subgoals.")

        outgoing = OutgoingMessage()
        outgoing.write_int32(int(step_index))
        outgoing.write_int32(int(area_index))
        for agent_index in range(3):
            outgoing.write_float32_list([float(value) for value in shared_translation[agent_index]])
            outgoing.write_float32_list([float(value) for value in formation_correction[agent_index]])
            outgoing.write_float32_list([float(value) for value in body_subgoal[agent_index]])
            outgoing.write_float32(float(yaw_error_deg[agent_index]))
            outgoing.write_int32(int(phase[agent_index]))
        self.queue_message_to_send(outgoing)


def attach_trinet_subgoal_debug_channel(env: Any, channel: TriNetSubgoalDebugChannel) -> None:
    """Expose the channel through wrapper layers without changing their API."""
    setattr(env, TRINET_SUBGOAL_DEBUG_CHANNEL_ATTR, channel)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:
        setattr(unwrapped, TRINET_SUBGOAL_DEBUG_CHANNEL_ATTR, channel)


def resolve_trinet_subgoal_debug_channel(env: Any) -> Optional[TriNetSubgoalDebugChannel]:
    if env is None:
        return None

    seen: set[int] = set()
    stack: list[Any] = [env]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        channel = getattr(current, TRINET_SUBGOAL_DEBUG_CHANNEL_ATTR, None)
        if isinstance(channel, TriNetSubgoalDebugChannel):
            return channel

        for attribute in ("env", "venv", "unwrapped"):
            child = getattr(current, attribute, None)
            if child is not None and child is not current:
                stack.append(child)

        children = getattr(current, "envs", None)
        if isinstance(children, (tuple, list)):
            stack.extend(children)

    return None


__all__ = [
    "TRINET_SUBGOAL_DEBUG_CHANNEL_ATTR",
    "TRINET_SUBGOAL_DEBUG_CHANNEL_ID",
    "TriNetSubgoalDebugChannel",
    "attach_trinet_subgoal_debug_channel",
    "resolve_trinet_subgoal_debug_channel",
]
