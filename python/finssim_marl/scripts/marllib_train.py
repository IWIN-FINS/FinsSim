"""FinsSim entrypoint for MARLlib-based Unity 3Chase1 benchmarks."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_marl.training.artifacts import resolve_training_output_paths


@dataclass
class MARLlibTrainArgs:
    """CLI arguments for MARLlib benchmark training."""

    config: str = "unity_3chase1"
    exp_name: str = "marllib_3chase1"
    output_dir: Optional[str] = None
    resolved_config: Optional[str] = None

    env_path: Optional[str] = None
    env_base_port: int = 5005
    num_envs: int = 1
    time_scale: float = 10.0
    show_graphics: bool = False
    seed: int = 42

    algorithm: str = "mappo"
    hyperparam_source: str = "common"
    map_name: str = "3Chase1"
    force_coop: bool = False
    share_policy: str = "individual"

    model_core_arch: str = "mlp"
    model_encode_layer: str = "128-128"

    stop_iters: int = 100
    stop_timesteps: int = 1_000_000
    stop_reward: int = 999_999
    episode_limit: int = 900

    local_mode: bool = False
    num_workers: Optional[int] = None
    num_gpus: int = 0
    checkpoint_freq: int = 50
    checkpoint_end: bool = True

    max_port_retries: int = 16
    prey_action_source: str = "wrapper_escape"


def _workspace_root() -> Path:
    current = PROJECT_ROOT.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").exists() and (candidate / "python" / "finssim_core").exists():
            return candidate
    return current


def _default_unity_binary() -> Path:
    return _workspace_root() / "artifacts/unity_builds/marl/linux/FinsROV/3Chase1_headless/3Chase1.x86_64"


def _load_resolved_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _write_summary(output_dir: Path, args: MARLlibTrainArgs, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = getattr(result, "metrics", [])
    summary = {
        "args": asdict(args),
        "ray_version": getattr(result, "ray_version", None),
        "last_result": getattr(result, "last_result", {}),
        "metrics": metrics,
        "note": (
            "This uses the current MARLlib Ray-2 compatibility runner. "
            "Verify MARLlib's runner is connected to a real RLlib learner before "
            "treating rewards as a definitive algorithm benchmark."
        ),
    }
    with (output_dir / "marllib_result.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)


def main() -> None:
    args = tyro.cli(MARLlibTrainArgs)

    random.seed(args.seed)
    np.random.seed(args.seed)

    from marllib import marl

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is None:
        class _ArtifactConfig:
            log_dir = "runs"
            checkpoint_dir = "checkpoints"
            tensorboard_dir = None
            output_dir = None

        run_name = f"{args.config}__{args.exp_name}"
        resolve_training_output_paths(_ArtifactConfig, run_name, backend="marllib")
        output_dir = Path(_ArtifactConfig.log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = _load_resolved_config(args.resolved_config)
    explicit_unity = resolved_config.get("unity", {}) if isinstance(resolved_config, dict) else {}

    unity_binary = args.env_path or explicit_unity.get("env_path") or str(_default_unity_binary())
    num_workers = args.num_workers if args.num_workers is not None else max(0, args.num_envs - 1)

    print("[FinsSim MARLlib] Starting benchmark")
    print(f"  algorithm: {args.algorithm}")
    print(f"  unity binary: {unity_binary}")
    print(f"  output dir: {output_dir}")
    print(
        "  runner note: current MARLlib Ray-2 compatibility runner must be upgraded "
        "for definitive learning benchmarks."
    )

    env = marl.make_env(
        environment_name="unity_3chase1",
        map_name=args.map_name,
        force_coop=args.force_coop,
        unity_env_binary_path=unity_binary,
        env_base_port=args.env_base_port,
        max_port_retries=args.max_port_retries,
        worker_id=0,
        no_graphics=not args.show_graphics,
        time_scale=args.time_scale,
        episode_limit=args.episode_limit,
        seed=args.seed,
        prey_action_source=args.prey_action_source,
    )

    algo_factory = getattr(marl.algos, args.algorithm)
    algo = algo_factory(hyperparam_source=args.hyperparam_source)
    model = marl.build_model(
        env,
        algo,
        {
            "core_arch": args.model_core_arch,
            "encode_layer": args.model_encode_layer,
        },
    )

    result = algo.fit(
        env,
        model,
        stop={
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,
        },
        local_mode=args.local_mode,
        num_gpus=args.num_gpus,
        num_workers=num_workers,
        share_policy=args.share_policy,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_end=args.checkpoint_end,
        local_dir=str(output_dir),
        seed=args.seed,
    )
    _write_summary(output_dir, args, result)


if __name__ == "__main__":
    main()
