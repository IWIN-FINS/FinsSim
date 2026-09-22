from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import signal
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _close_env_safely(env: gym.Env | None) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:
        print(f"[ManagedSubprocVecEnv] env.close() failed during cleanup: {exc}")


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    try:
        os.setsid()
    except Exception:
        pass

    parent_remote.close()
    env: gym.Env | None = None
    reset_info: Optional[dict[str, Any]] = {}
    cleaned_up = False

    def _cleanup() -> None:
        nonlocal env, cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        _close_env_safely(env)
        env = None
        try:
            remote.close()
        except Exception:
            pass
        try:
            parent_remote.close()
        except Exception:
            pass

    def _handle_signal(signum, _frame) -> None:
        _cleanup()
        raise SystemExit(128 + int(signum))

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        env = _patch_env(env_fn_wrapper.var())
        while True:
            try:
                cmd, data = remote.recv()
                if cmd == "step":
                    observation, reward, terminated, truncated, info = env.step(data)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:
                        info["terminal_observation"] = observation
                        observation, reset_info = env.reset()
                    remote.send((observation, reward, done, info, reset_info))
                elif cmd == "reset":
                    maybe_options = {"options": data[1]} if data[1] else {}
                    observation, reset_info = env.reset(seed=data[0], **maybe_options)
                    remote.send((observation, reset_info))
                elif cmd == "render":
                    remote.send(env.render())
                elif cmd == "close":
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                elif cmd == "env_method":
                    method = env.get_wrapper_attr(data[0])
                    remote.send(method(*data[1], **data[2]))
                elif cmd == "send_high_level_target_debug":
                    from finssim_rl.training.high_level_target_debug_channel import (
                        resolve_high_level_target_debug_channel,
                    )

                    channel = resolve_high_level_target_debug_channel(env)
                    if channel is None:
                        remote.send(False)
                    else:
                        channel.send_high_level_target(
                            step_index=data["step_index"],
                            local_target_delta=data["local_target_delta"],
                            target_world=data["target_world"],
                            current_world=data["current_world"],
                        )
                        remote.send(True)
                elif cmd == "get_attr":
                    remote.send(env.get_wrapper_attr(data))
                elif cmd == "has_attr":
                    try:
                        env.get_wrapper_attr(data)
                        remote.send(True)
                    except AttributeError:
                        remote.send(False)
                elif cmd == "set_attr":
                    remote.send(setattr(env, data[0], data[1]))
                elif cmd == "is_wrapped":
                    remote.send(is_wrapped(env, data))
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    finally:
        _cleanup()


class ManagedSubprocVecEnv(VecEnv):
    """SB3-compatible SubprocVecEnv with stronger shutdown behavior for Unity."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=False)
            process.start()
            self.processes.append(process)
            work_remote.close()

        try:
            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self.remotes[0].recv()
            super().__init__(len(env_fns), observation_space, action_space)
        except BaseException:
            self.close()
            raise

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return

        try:
            self.waiting = False

            for remote in self.remotes:
                try:
                    remote.send(("close", None))
                except Exception:
                    pass

            for process in self.processes:
                try:
                    process.join(timeout=5.0)
                except Exception:
                    pass
                if process.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except Exception:
                        process.terminate()
                    process.join(timeout=2.0)
                if process.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        process.kill()
                    process.join(timeout=1.0)
        finally:
            for remote in self.remotes:
                try:
                    remote.close()
                except Exception:
                    pass
            self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def has_attr(self, attr_name: str) -> bool:
        target_remotes = self._get_target_remotes(indices=None)
        for remote in target_remotes:
            remote.send(("has_attr", attr_name))
        return all([remote.recv() for remote in target_remotes])

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def send_high_level_target_debug(
        self,
        *,
        step_index: int,
        local_target_delta: np.ndarray,
        target_world: np.ndarray,
        current_world: np.ndarray,
    ) -> list[bool]:
        local_target_delta = np.asarray(local_target_delta, dtype=np.float32)
        target_world = np.asarray(target_world, dtype=np.float32)
        current_world = np.asarray(current_world, dtype=np.float32)
        if local_target_delta.ndim == 1:
            local_target_delta = local_target_delta.reshape(1, -1)
        if target_world.ndim == 1:
            target_world = target_world.reshape(1, -1)
        if current_world.ndim == 1:
            current_world = current_world.reshape(1, -1)

        count = min(
            len(self.remotes),
            local_target_delta.shape[0],
            target_world.shape[0],
            current_world.shape[0],
        )
        for env_idx, remote in enumerate(self.remotes[:count]):
            remote.send(
                (
                    "send_high_level_target_debug",
                    {
                        "step_index": int(step_index) + env_idx,
                        "local_target_delta": local_target_delta[env_idx, :3].tolist(),
                        "target_world": target_world[env_idx, :3].tolist(),
                        "current_world": current_world[env_idx, :3].tolist(),
                    },
                )
            )
        return [remote.recv() for remote in self.remotes[:count]]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _stack_obs(obs_list: Union[list[VecEnvObs], tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs_list) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs_list[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.stack([single_obs[key] for single_obs in obs_list]) for key in space.spaces.keys()}
    if isinstance(space, spaces.Tuple):
        assert isinstance(obs_list[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([single_obs[i] for single_obs in obs_list]) for i in range(obs_len))
    return np.stack(obs_list)
