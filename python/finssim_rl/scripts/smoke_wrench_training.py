"""Minimal smoke runner for PPO_WRENCH Unity builds.

This intentionally bypasses the normal long training config and runs one tiny
PPO update.  It is for checking Unity startup, action dimensions, and the
6D-policy-to-8D-simulator adapter.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_rl.models import make_unity_training_areas_env
from finssim_rl.training.config import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal PPO_WRENCH smoke training pass.")
    parser.add_argument("--name", required=True, help="Smoke run name used under artifacts.")
    parser.add_argument("--env-path", required=True, help="Path to the Unity executable.")
    parser.add_argument("--env-base-port", type=int, required=True, help="Unity base port.")
    parser.add_argument("--device", default="cpu", help="Training device for the smoke model.")
    parser.add_argument("--timesteps", type=int, default=16, help="Tiny total_timesteps for smoke.")
    parser.add_argument("--timeout-wait", type=int, default=240, help="Unity startup timeout.")
    parser.add_argument("--num-areas", type=int, default=1, help="Replicated ML-Agents areas in one Unity binary.")
    parser.add_argument(
        "--output-root",
        default="./artifacts/runs/rl/wrench/smoke",
        help="Directory where smoke TensorBoard logs are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_path).expanduser().resolve()
    if not env_path.exists():
        print(f"SMOKE FAIL: env executable does not exist: {env_path}", flush=True)
        return 2

    print(f"===== SMOKE {args.name} =====", flush=True)
    print(f"Unity env: {env_path}", flush=True)
    print(f"Base port: {args.env_base_port}", flush=True)

    base_config = get_config("ppo_wrench_for_pose_empirical_thruster_mixer")
    env_config = dataclasses.replace(
        base_config.env_config,
        env_path=str(env_path),
        # ``num_envs`` is the canonical configuration field. In multi-area
        # mode it denotes the number of replicated Unity training areas.
        num_envs=max(1, int(args.num_areas)),
        eval_num_envs=1,
        env_base_port=int(args.env_base_port),
        port_offset=0,
        time_scale=10.0,
        no_graphics=True,
        timeout_wait=int(args.timeout_wait),
    )
    training_config = dataclasses.replace(
        base_config.training_config,
        n_steps=int(args.timesteps),
        batch_size=int(args.timesteps),
        n_epochs=1,
        total_timesteps=int(args.timesteps),
        checkpoint_freq=int(args.timesteps),
        eval_freq=1_000_000,
    )
    config = dataclasses.replace(
        base_config,
        env_config=env_config,
        training_config=training_config,
    )

    smoke_dir = Path(args.output_root).expanduser().resolve() / args.name
    shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    model_args = SimpleNamespace(device=args.device, tensorboard_dir=str(smoke_dir / "tensorboard"))

    env = make_unity_training_areas_env(str(env_path), env_config)
    try:
        print(f"Observation Space: {env.observation_space}", flush=True)
        print(f"Action Space: {env.action_space}", flush=True)
        model = config.create_model(env=env, args=model_args, load_checkpoint=None)
        policy_dim = int(model.policy.action_net.out_features)
        env_dim = int(env.action_space.shape[0])
        print(f"Policy action dim: {policy_dim}; env action dim: {env_dim}", flush=True)
        if policy_dim != 6 or env_dim != 8:
            print("SMOKE FAIL: expected policy_dim=6 and env_dim=8", flush=True)
            return 3

        model.learn(total_timesteps=training_config.total_timesteps, progress_bar=False)
        print(f"SMOKE PASS: {args.name}", flush=True)
        return 0
    finally:
        print(f"Closing env: {args.name}", flush=True)
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
