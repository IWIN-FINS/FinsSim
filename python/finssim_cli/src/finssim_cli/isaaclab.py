"""Isaac Lab train/play launcher used by the top-level FinsSim CLI.

The module deliberately does not reuse the Unity experiment schema.  Isaac Lab
owns simulation and training; this layer only validates a small launch YAML,
starts the correct Isaac Python environment, and records a portable index.
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml


RLLibrary = Literal["rsl_rl", "skrl"]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _resolve_path(value: str | None, *, config_path: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(value)).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class IsaacLabExperiment:
    name: str
    seed: int = 42
    tags: tuple[str, ...] = ()
    output_dir: Path | None = None


@dataclass(frozen=True)
class IsaacLabLaunch:
    path: Path
    project_path: Path
    task: str
    rl_library: RLLibrary
    cuda_visible_devices: str | None = None
    device: str | None = None
    num_envs: int = 1
    max_iterations: int | None = None
    headless: bool = True
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    hydra_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class IsaacLabTrain:
    resume_checkpoint: Path | None = None


@dataclass(frozen=True)
class IsaacLabPlay:
    checkpoint: Path | None = None
    num_envs: int = 1
    headless: bool = False
    real_time: bool = False
    video: bool = False


@dataclass(frozen=True)
class IsaacLabConfig:
    path: Path
    experiment: IsaacLabExperiment
    isaaclab: IsaacLabLaunch
    train: IsaacLabTrain
    play: IsaacLabPlay
    raw: dict[str, Any]


def load_isaaclab_config(path: str | Path, *, workspace_root: Path) -> IsaacLabConfig:
    """Load the intentionally separate Isaac Lab experiment YAML."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    raw = _mapping(raw, "IsaacLab config")
    unknown = sorted(set(raw) - {"experiment", "isaaclab", "train", "play"})
    if unknown:
        raise ValueError(f"Unknown IsaacLab config sections: {', '.join(unknown)}")

    experiment_raw = _mapping(raw.get("experiment"), "experiment")
    if not experiment_raw.get("name"):
        raise ValueError("experiment.name is required")
    tags = experiment_raw.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError("experiment.tags must be a list")
    experiment_unknown = sorted(set(experiment_raw) - {"name", "seed", "tags", "output_dir"})
    if experiment_unknown:
        raise ValueError(f"Unknown experiment fields: {', '.join(experiment_unknown)}")
    output_dir = _resolve_path(experiment_raw.get("output_dir"), config_path=config_path)
    experiment = IsaacLabExperiment(
        name=str(experiment_raw["name"]),
        seed=int(experiment_raw.get("seed", 42)),
        tags=tuple(str(tag) for tag in tags),
        output_dir=output_dir,
    )

    launch_raw = _mapping(raw.get("isaaclab"), "isaaclab")
    launch_unknown = sorted(
        set(launch_raw)
        - {
            "path",
            "project_path",
            "task",
            "rl_library",
            "cuda_visible_devices",
            "device",
            "num_envs",
            "max_iterations",
            "headless",
            "video",
            "video_length",
            "video_interval",
            "hydra_overrides",
        }
    )
    if launch_unknown:
        raise ValueError(f"Unknown isaaclab fields: {', '.join(launch_unknown)}")
    if not launch_raw.get("task"):
        raise ValueError("isaaclab.task is required")
    rl_library = str(launch_raw.get("rl_library", "rsl_rl"))
    if rl_library not in {"rsl_rl", "skrl"}:
        raise ValueError("isaaclab.rl_library must be 'rsl_rl' or 'skrl'")
    hydra_overrides = launch_raw.get("hydra_overrides", [])
    if not isinstance(hydra_overrides, list) or not all(isinstance(item, str) for item in hydra_overrides):
        raise ValueError("isaaclab.hydra_overrides must be a list of strings")
    isaaclab_path = _resolve_path(launch_raw.get("path"), config_path=config_path)
    if isaaclab_path is None:
        configured = os.environ.get("ISAACLAB_PATH", "").strip()
        if not configured:
            raise ValueError("isaaclab.path is required; set it in the YAML or export ISAACLAB_PATH")
        isaaclab_path = Path(configured).expanduser()
    project_path = _resolve_path(launch_raw.get("project_path"), config_path=config_path)
    if project_path is None:
        project_path = workspace_root / "simulators/isaaclab/FinsSimIsaacLab"
    num_envs = int(launch_raw.get("num_envs", 1))
    if num_envs < 1:
        raise ValueError("isaaclab.num_envs must be positive")
    max_iterations_raw = launch_raw.get("max_iterations")
    max_iterations = None if max_iterations_raw is None else int(max_iterations_raw)
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("isaaclab.max_iterations must be positive when provided")
    launch = IsaacLabLaunch(
        path=isaaclab_path.resolve(),
        project_path=project_path.resolve(),
        task=str(launch_raw["task"]),
        rl_library=rl_library,  # type: ignore[arg-type]
        cuda_visible_devices=(
            None if launch_raw.get("cuda_visible_devices") is None else str(launch_raw["cuda_visible_devices"])
        ),
        device=None if launch_raw.get("device") is None else str(launch_raw["device"]),
        num_envs=num_envs,
        max_iterations=max_iterations,
        headless=bool(launch_raw.get("headless", True)),
        video=bool(launch_raw.get("video", False)),
        video_length=int(launch_raw.get("video_length", 200)),
        video_interval=int(launch_raw.get("video_interval", 2000)),
        hydra_overrides=tuple(hydra_overrides),
    )

    train_raw = raw.get("train") or {}
    train_raw = _mapping(train_raw, "train")
    unknown_train = sorted(set(train_raw) - {"resume_checkpoint"})
    if unknown_train:
        raise ValueError(f"Unknown train fields: {', '.join(unknown_train)}")
    train = IsaacLabTrain(_resolve_path(train_raw.get("resume_checkpoint"), config_path=config_path))

    play_raw = raw.get("play") or {}
    play_raw = _mapping(play_raw, "play")
    unknown_play = sorted(set(play_raw) - {"checkpoint", "num_envs", "headless", "real_time", "video"})
    if unknown_play:
        raise ValueError(f"Unknown play fields: {', '.join(unknown_play)}")
    play_num_envs = int(play_raw.get("num_envs", 1))
    if play_num_envs < 1:
        raise ValueError("play.num_envs must be positive")
    play = IsaacLabPlay(
        checkpoint=_resolve_path(play_raw.get("checkpoint"), config_path=config_path),
        num_envs=play_num_envs,
        headless=bool(play_raw.get("headless", False)),
        real_time=bool(play_raw.get("real_time", False)),
        video=bool(play_raw.get("video", False)),
    )
    return IsaacLabConfig(config_path, experiment, launch, train, play, raw)


def _backend_log_root(config: IsaacLabConfig) -> Path:
    return config.isaaclab.path / "logs" / config.isaaclab.rl_library


def _validate_checkpoint(checkpoint: Path, config: IsaacLabConfig) -> Path:
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist or is not a file: {checkpoint}")
    expected_root = _backend_log_root(config).resolve()
    try:
        checkpoint.resolve().relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(
            f"{config.isaaclab.rl_library} checkpoint must be below {expected_root}; "
            "cross-backend checkpoints are not compatible"
        ) from exc
    return checkpoint.resolve()


def _checkpoint_run_dir(checkpoint: Path, rl_library: RLLibrary) -> Path:
    """Return the native run directory containing a backend checkpoint."""
    if rl_library == "skrl" and checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _validate_runtime(config: IsaacLabConfig) -> None:
    launcher = config.isaaclab.path / "isaaclab.sh"
    python_exe = config.isaaclab.path / "env_isaaclab" / "bin" / "python"
    extension = config.isaaclab.project_path / "source" / "finssim_isaaclab_tasks"
    if not launcher.is_file():
        raise ValueError(f"Isaac Lab launcher does not exist: {launcher}")
    if not python_exe.is_file():
        raise ValueError(f"Isaac Lab Python does not exist: {python_exe}")
    if not extension.is_dir():
        raise ValueError(f"FinsSim Isaac task extension does not exist: {extension}")
    if config.isaaclab.rl_library == "skrl":
        probe = subprocess.run(
            [str(python_exe), "-c", "import skrl; assert tuple(map(int, skrl.__version__.split('.')[:2])) >= (2, 1)"],
            env=_child_environment(config),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            requirements = config.isaaclab.project_path / "requirements-isaaclab.txt"
            install = (
                f"cd {shlex.quote(str(config.isaaclab.path))} && "
                "env -u VIRTUAL_ENV -u CONDA_PREFIX ./isaaclab.sh -p -m pip install -r "
                f"{shlex.quote(str(requirements))}"
            )
            raise ValueError(f"SKRL >= 2.1.0 is unavailable in IsaacLab Python. Install it with:\n{install}")


def _child_environment(config: IsaacLabConfig) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME"):
        env.pop(key, None)
    extension = str(config.isaaclab.project_path / "source" / "finssim_isaaclab_tasks")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extension if not existing_pythonpath else f"{extension}{os.pathsep}{existing_pythonpath}"
    env["ISAACLAB_PATH"] = str(config.isaaclab.path)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if config.isaaclab.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = config.isaaclab.cuda_visible_devices
    return env


def _run_id() -> str:
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{secrets.token_hex(4)}"


def _run_dir(config: IsaacLabConfig, workspace_root: Path, run_id: str) -> Path:
    root = config.experiment.output_dir or (workspace_root / "artifacts/runs/isaaclab" / config.experiment.name)
    return root / run_id


def _wrapper(config: IsaacLabConfig, action: Literal["train", "play"]) -> Path:
    suffix = "" if config.isaaclab.rl_library == "rsl_rl" else "_skrl"
    return config.isaaclab.project_path / "scripts" / f"{action}{suffix}.py"


def _build_command(
    config: IsaacLabConfig,
    *,
    action: Literal["train", "play"],
    run_id: str,
) -> list[str]:
    wrapper = _wrapper(config, action)
    if not wrapper.is_file():
        raise ValueError(f"FinsSim {config.isaaclab.rl_library} {action} wrapper does not exist: {wrapper}")
    launch = config.isaaclab
    command = [str(launch.path / "isaaclab.sh"), "-p", str(wrapper), "--task", launch.task]
    if action == "train":
        command.extend(["--num_envs", str(launch.num_envs), "--seed", str(config.experiment.seed)])
        if launch.max_iterations is not None:
            command.extend(["--max_iterations", str(launch.max_iterations)])
        if launch.video:
            command.extend(["--video", "--video_length", str(launch.video_length), "--video_interval", str(launch.video_interval)])
        if launch.rl_library == "rsl_rl":
            command.extend(["--run_name", run_id])
            if config.train.resume_checkpoint is not None:
                checkpoint = _validate_checkpoint(config.train.resume_checkpoint, config)
                command.extend(["--resume", "--load_run", checkpoint.parent.name, "--checkpoint", checkpoint.name])
        elif config.train.resume_checkpoint is not None:
            checkpoint = _validate_checkpoint(config.train.resume_checkpoint, config)
            command.extend(["--checkpoint", str(checkpoint)])
    else:
        if config.play.checkpoint is None:
            raise ValueError("play.checkpoint is required for 'finssim isaaclab play'")
        checkpoint = _validate_checkpoint(config.play.checkpoint, config)
        command.extend(["--num_envs", str(config.play.num_envs), "--seed", str(config.experiment.seed), "--checkpoint", str(checkpoint)])
        if config.play.real_time:
            command.append("--real-time")
        if config.play.video:
            command.extend(["--video", "--video_length", str(launch.video_length)])
    headless = launch.headless if action == "train" else config.play.headless
    if headless:
        # Isaac Lab 3 keeps --headless for compatibility but recommends the
        # explicit visualization preset instead.
        command.extend(["--viz", "none"])
    else:
        # Do not rely on the upstream default: visual replay must always open
        # Kit when the YAML requests non-headless operation.
        command.extend(["--viz", "kit"])
    if launch.device:
        command.extend(["--device", launch.device])
    command.extend(launch.hydra_overrides)
    return command


_LOG_ROOT = re.compile(r"^\[INFO\] (?:Logging experiment|Loading experiment) (?:in|from) directory: (.+)$")
_EXACT_RUN = re.compile(r"^Exact experiment name requested from command line: (.+)$")


def _manifest_payload(
    config: IsaacLabConfig,
    *,
    run_id: str,
    action: str,
    command: list[str],
    status: str,
    native_log_dir: Path | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "action": action,
        "command": command,
        "native_log_dir": str(native_log_dir) if native_log_dir else None,
        "returncode": returncode,
        "experiment": {
            "name": config.experiment.name,
            "seed": config.experiment.seed,
            "tags": list(config.experiment.tags),
        },
        "isaaclab": asdict(config.isaaclab),
        "config_path": str(config.path),
    }
    payload["isaaclab"]["path"] = str(config.isaaclab.path)
    payload["isaaclab"]["project_path"] = str(config.isaaclab.project_path)
    return payload


def _write_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    with (run_dir / "manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)


def launch_isaaclab(
    config_path: Path,
    *,
    workspace_root: Path,
    action: Literal["train", "play"],
    dry_run: bool = False,
) -> None:
    """Validate, launch, and index an Isaac Lab task run."""
    config = load_isaaclab_config(config_path, workspace_root=workspace_root)
    _validate_runtime(config)
    run_id = _run_id()
    command = _build_command(config, action=action, run_id=run_id)
    run_dir = _run_dir(config, workspace_root, run_id)
    if dry_run:
        print(f"Backend cwd: {config.isaaclab.path}")
        print(f"FinsSim index: {run_dir}")
        print(f"Command: {shlex.join(command)}")
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.raw, stream, allow_unicode=True, sort_keys=False)
    native_log_dir: Path | None = None
    if action == "play":
        # _build_command already validates this field; record the exact source
        # run rather than only the backend log root in the FinsSim index.
        assert config.play.checkpoint is not None
        native_log_dir = _checkpoint_run_dir(
            _validate_checkpoint(config.play.checkpoint, config), config.isaaclab.rl_library
        )
    _write_manifest(
        run_dir,
        _manifest_payload(
            config,
            run_id=run_id,
            action=action,
            command=command,
            status="started",
            native_log_dir=native_log_dir,
        ),
    )

    proc = subprocess.Popen(
        command,
        cwd=config.isaaclab.path,
        env=_child_environment(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_root: Path | None = None
    exact_run: str | None = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stripped = line.rstrip("\n")
            root_match = _LOG_ROOT.match(stripped)
            if root_match:
                log_root = Path(root_match.group(1))
            exact_match = _EXACT_RUN.match(stripped)
            if exact_match:
                exact_run = exact_match.group(1)
            if log_root and exact_run and action == "train":
                suffix = f"_{run_id}" if config.isaaclab.rl_library == "rsl_rl" else ""
                native_log_dir = log_root / f"{exact_run}{suffix}"
                _write_manifest(
                    run_dir,
                    _manifest_payload(
                        config,
                        run_id=run_id,
                        action=action,
                        command=command,
                        status="running",
                        native_log_dir=native_log_dir,
                    ),
                )
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
        _write_manifest(
            run_dir,
            _manifest_payload(
                config,
                run_id=run_id,
                action=action,
                command=command,
                status="interrupted",
                native_log_dir=native_log_dir,
                returncode=returncode,
            ),
        )
        raise
    status = "completed" if returncode == 0 else "failed"
    _write_manifest(
        run_dir,
        _manifest_payload(
            config,
            run_id=run_id,
            action=action,
            command=command,
            status=status,
            native_log_dir=native_log_dir,
            returncode=returncode,
        ),
    )
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
