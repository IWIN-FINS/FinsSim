"""FinsSim entrypoint for BenchMARL/TorchRL-based MARL experiments."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tyro
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_marl.integrations.benchmarl.experiment import build_task
from finssim_marl.training.artifacts import resolve_training_output_paths


@dataclass
class BenchMARLTrainArgs:
    """CLI arguments for BenchMARL integration smoke and training setup."""

    config: str = "chasing_3chase1"
    exp_name: str = "benchmarl_chasing_3chase1"
    output_dir: Optional[str] = None
    resolved_config: Optional[str] = None

    env_path: Optional[str] = None
    env_base_port: int = 5005
    num_envs: int = 1
    time_scale: float = 10.0
    show_graphics: bool = False
    seed: int = 42
    worker_id: int = 0
    episode_limit: int = 900
    prey_action_source: str = "wrapper_escape"

    dry_run: bool = False
    check_env_specs: bool = False


def _load_resolved_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _resolve_output_dir(args: BenchMARLTrainArgs) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()

    class _ArtifactConfig:
        log_dir = "runs"
        checkpoint_dir = "checkpoints"
        tensorboard_dir = None
        output_dir = None

    resolve_training_output_paths(
        _ArtifactConfig,
        f"{args.config}__{args.exp_name}",
        backend="marl",
    )
    return Path(_ArtifactConfig.log_dir).resolve()


def _task_config(args: BenchMARLTrainArgs, resolved_config: dict) -> dict:
    explicit_unity = resolved_config.get("unity", {}) if isinstance(resolved_config, dict) else {}
    env_path = args.env_path or explicit_unity.get("env_path")
    return {
        "env_type": "3chase1",
        "unity_env_binary_path": env_path or "",
        "env_base_port": args.env_base_port,
        "worker_id": args.worker_id,
        "time_scale": args.time_scale,
        "no_graphics": not args.show_graphics,
        "episode_limit": args.episode_limit,
        "prey_action_source": args.prey_action_source,
    }


def main() -> None:
    args = tyro.cli(BenchMARLTrainArgs)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _load_resolved_config(args.resolved_config)
    task_config = _task_config(args, resolved_config)
    task = build_task(task_config)

    summary = {
        "args": asdict(args),
        "task": str(task),
        "task_config": task_config,
        "note": (
            "BenchMARL integration is wired through FinsSimTorchRLEnv. "
            "Use --check-env-specs only when a valid Unity binary is available."
        ),
    }

    if args.check_env_specs:
        from torchrl.envs import check_env_specs

        env = task.get_env_fun(
            num_envs=args.num_envs,
            continuous_actions=True,
            seed=args.seed,
            device="cpu",
        )()
        try:
            check_env_specs(env)
            summary["check_env_specs"] = "ok"
        finally:
            env.close()

    with (output_dir / "benchmarl_setup.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print("[FinsSim BenchMARL] Setup complete")
    print(f"  output dir: {output_dir}")
    print(f"  task: {task}")
    if args.dry_run:
        print("  dry-run: no BenchMARL training launched")


if __name__ == "__main__":
    main()
