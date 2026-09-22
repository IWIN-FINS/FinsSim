"""SB3 VecEnv adapter for ML-Agents TrainingAreaReplicator builds.

ML-Agents exposes every replicated training area as an Agent under the same
behavior.  ``UnityToGymWrapper`` intentionally rejects that arrangement; this
adapter maps those agent ids to Stable-Baselines3 vector-environment slots.
"""

from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


class UnityTrainingAreasVecEnv(VecEnv):
    """Expose one ML-Agents behavior with ``num_areas`` agents as an SB3 VecEnv.

    The target scene must use one homogeneous behavior and request a decision
    for every area at every public environment step.  HoldForPosition's
    parallel scene explicitly requests a decision after an episode reset so a
    terminal area is available again in the same response.
    """

    # A normal VecEnv step must receive the next decision promptly. This is
    # intentionally separate from the wall-clock startup readiness barrier.
    _STEP_RESPONSE_MAX_STEPS = 32

    def __init__(
        self,
        unity_env: Any,
        *,
        num_areas: int,
        uint8_visual: bool = False,
        allow_multiple_obs: bool = False,
        ready_timeout_seconds: float = 60.0,
    ) -> None:
        if num_areas < 1:
            raise ValueError("num_areas must be positive")
        self.unity_env = unity_env
        self.num_areas_requested = int(num_areas)
        self.uint8_visual = bool(uint8_visual)
        self.allow_multiple_obs = bool(allow_multiple_obs)
        self.ready_timeout_seconds = max(0.0, float(ready_timeout_seconds))
        self._pending_actions: np.ndarray | None = None
        self._closed = False

        if not self.unity_env.behavior_specs:
            self.unity_env.step()
        if len(self.unity_env.behavior_specs) != 1:
            raise ValueError("Training-area VecEnv requires exactly one ML-Agents behavior")
        self.behavior_name = next(iter(self.unity_env.behavior_specs))
        self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
        action_spec = self.behavior_spec.action_spec
        if not action_spec.is_continuous():
            raise ValueError("Training-area VecEnv currently requires continuous ML-Agents actions")

        self._observation_space = self._make_observation_space()
        action_size = int(action_spec.continuous_size)
        self._action_space = spaces.Box(-1.0, 1.0, shape=(action_size,), dtype=np.float32)

        self._slot_agent_ids: np.ndarray | None = None
        self._agent_to_slot: dict[int, int] = {}
        self._current_decision_agent_ids = np.empty(0, dtype=np.int64)
        super().__init__(self.num_areas_requested, self._observation_space, self._action_space)
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._last_obs = self._reset_to_ready_observation()

    def _make_observation_space(self) -> spaces.Space:
        specs = list(self.behavior_spec.observation_specs)
        if not specs:
            raise ValueError("ML-Agents behavior has no observations")

        def make_space(spec: Any) -> spaces.Box:
            shape = tuple(int(value) for value in spec.shape)
            if self.uint8_visual and len(shape) == 3:
                return spaces.Box(0, 255, shape=shape, dtype=np.uint8)
            return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

        if self.allow_multiple_obs:
            return spaces.Tuple(tuple(make_space(spec) for spec in specs))
        return make_space(specs[0])

    def _set_agent_slots(self, agent_ids: Iterable[int]) -> None:
        ids = [int(agent_id) for agent_id in agent_ids]
        if len(ids) != self.num_areas_requested:
            raise RuntimeError(
                f"Unity reported {len(ids)} active agents for behavior {self.behavior_name!r}; "
                f"expected num_areas={self.num_areas_requested}. "
                "Check TrainingAreaReplicator and the Python --num-areas value."
            )
        if len(set(ids)) != len(ids):
            raise RuntimeError("Unity reported duplicate ML-Agents agent ids")
        self._slot_agent_ids = np.asarray(ids, dtype=np.int64)
        self._agent_to_slot = {agent_id: slot for slot, agent_id in enumerate(ids)}
        self._current_decision_agent_ids = self._slot_agent_ids.copy()

    def _ensure_agent_slots(self, agent_ids: Iterable[int]) -> None:
        ids = [int(agent_id) for agent_id in agent_ids]
        if self._slot_agent_ids is None:
            self._set_agent_slots(ids)
            return
        if set(ids) != set(int(agent_id) for agent_id in self._slot_agent_ids):
            raise RuntimeError("Unity changed the TrainingArea agent-id set during reset")

    def _ordered_indices(self, steps: Any) -> np.ndarray:
        try:
            return np.asarray([self._agent_to_slot[int(agent_id)] for agent_id in steps.agent_id], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"Unity returned an unknown agent id: {exc.args[0]}") from exc

    def _advance_reset_warmup(self, decision_steps: Any) -> None:
        """Advance one initial physics tick without exposing a partial VecEnv step."""
        if len(decision_steps):
            from mlagents_envs.base_env import ActionTuple

            actions = ActionTuple()
            actions.add_continuous(
                np.zeros((len(decision_steps), self.action_space.shape[0]), dtype=np.float32)
            )
            self.unity_env.set_actions(self.behavior_name, actions)
        self.unity_env.step()

    def _reset_to_ready_observation(self) -> VecEnvObs:
        """Return a full non-terminal observation after Unity's reset settling ticks.

        Some physics-heavy replicated scenes emit terminal records from the
        just-reset episode while their copies settle onto their local water
        plane.  Those records are reset bookkeeping, not SB3 transitions.
        Advance with zeros until all slots have a fresh decision together.

        A TrainingAreaReplicator can expose the base agent before every clone
        has registered with ML-Agents.  This is a startup barrier, so it is
        bounded by wall-clock time rather than an arbitrary number of physics
        ticks: clone creation time scales with the scene and area count.
        """
        self.unity_env.reset()
        observed_batch_sizes: list[tuple[int, int]] = []
        deadline = time.monotonic() + self.ready_timeout_seconds
        while True:
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
            observed_batch_sizes.append((len(decision_steps), len(terminal_steps)))
            if len(decision_steps) == self.num_areas_requested:
                self._ensure_agent_slots(decision_steps.agent_id)
                if not len(terminal_steps):
                    self._current_decision_agent_ids = np.asarray(decision_steps.agent_id, dtype=np.int64)
                    return self._obs_from_steps(decision_steps)
            if time.monotonic() >= deadline:
                break
            self._advance_reset_warmup(decision_steps)
        raise RuntimeError(
            f"Unity did not produce one clean decision batch for behavior {self.behavior_name!r}; "
            f"expected num_areas={self.num_areas_requested}, observed "
            f"{len(observed_batch_sizes)} (decision, terminal) batches; last batches="
            f"{observed_batch_sizes[-16:]}; readiness timeout={self.ready_timeout_seconds:.1f}s. "
            "Check TrainingAreaReplicator and the Python --num-areas value."
        )

    def _convert_obs(self, observation: np.ndarray) -> np.ndarray:
        if self.uint8_visual and observation.ndim == 4:
            return np.clip(np.rint(observation * 255.0), 0, 255).astype(np.uint8)
        return np.asarray(observation, dtype=np.float32)

    def _obs_from_steps(self, steps: Any) -> VecEnvObs:
        indices = self._ordered_indices(steps)
        if self.allow_multiple_obs:
            result: list[np.ndarray] = []
            for raw in steps.obs:
                ordered = np.empty((self.num_envs, *raw.shape[1:]), dtype=self._convert_obs(raw).dtype)
                ordered[indices] = self._convert_obs(raw)
                result.append(ordered)
            return tuple(result)
        raw = self._convert_obs(steps.obs[0])
        ordered = np.empty((self.num_envs, *raw.shape[1:]), dtype=raw.dtype)
        ordered[indices] = raw
        return ordered

    def _terminal_obs_by_slot(self, terminal_steps: Any) -> dict[int, Any]:
        if not len(terminal_steps):
            return {}
        slots = self._ordered_indices(terminal_steps)
        if self.allow_multiple_obs:
            return {
                int(slot): tuple(self._convert_obs(raw)[index].copy() for raw in terminal_steps.obs)
                for index, slot in enumerate(slots)
            }
        raw = self._convert_obs(terminal_steps.obs[0])
        return {int(slot): raw[index].copy() for index, slot in enumerate(slots)}

    def _merge_decision_observations(self, previous: VecEnvObs, steps: Any) -> VecEnvObs:
        slots = self._ordered_indices(steps)
        if self.allow_multiple_obs:
            merged = [value.copy() for value in previous]
            for index, raw in enumerate(steps.obs):
                merged[index][slots] = self._convert_obs(raw)
            return tuple(merged)
        merged = previous.copy()
        merged[slots] = self._convert_obs(steps.obs[0])
        return merged

    def reset(self) -> VecEnvObs:
        self._last_obs = self._reset_to_ready_observation()
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self._last_obs

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, *self.action_space.shape)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {actions.shape}")
        self._pending_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_actions is None:
            raise RuntimeError("step_wait() called before step_async()")
        from mlagents_envs.base_env import ActionTuple

        current_slots = np.asarray(
            [self._agent_to_slot[int(agent_id)] for agent_id in self._current_decision_agent_ids], dtype=np.int64
        )
        action_tuple = ActionTuple()
        action_tuple.add_continuous(self._pending_actions[current_slots])
        self.unity_env.set_actions(self.behavior_name, action_tuple)
        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        current_obs = self._last_obs.copy() if not self.allow_multiple_obs else tuple(value.copy() for value in self._last_obs)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        observed_slots: set[int] = set()
        for _ in range(self._STEP_RESPONSE_MAX_STEPS):
            decision_slots = self._ordered_indices(decision_steps)
            current_obs = self._merge_decision_observations(current_obs, decision_steps)
            for index, slot in enumerate(decision_slots):
                slot = int(slot)
                if slot not in observed_slots:
                    rewards[slot] = float(decision_steps.reward[index])
                    observed_slots.add(slot)
            terminal_observations = self._terminal_obs_by_slot(terminal_steps)
            for index, slot in enumerate(self._ordered_indices(terminal_steps)):
                slot = int(slot)
                rewards[slot] = float(terminal_steps.reward[index])
                dones[slot] = True
                infos[slot]["terminal_observation"] = terminal_observations[slot]
                infos[slot]["TimeLimit.truncated"] = bool(terminal_steps.interrupted[index])
            if len(observed_slots) == self.num_envs:
                self._current_decision_agent_ids = np.asarray(decision_steps.agent_id, dtype=np.int64)
                break
            self._advance_reset_warmup(decision_steps)
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        else:
            raise RuntimeError("Unity did not return a next decision for every TrainingArea within the step budget")

        self._pending_actions = None
        self._last_obs = current_obs
        return current_obs, rewards, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.unity_env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for _ in self._get_indices(indices):
            setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any) -> list[Any]:
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
