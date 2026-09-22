from __future__ import annotations

import dataclasses
import json
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from finssim_core.artifacts import default_artifacts_root
from finssim_core.config_loader import FinsSimExperimentConfig, load_experiment_config
from finssim_core.metadata import write_run_metadata
from finssim_cli.isaaclab import launch_isaaclab

app = typer.Typer(help="FinsSim platform command line tools.")
rl_app = typer.Typer(help="Single-agent RL backend commands.")
marl_app = typer.Typer(help="Multi-agent RL backend commands.")


@dataclass(frozen=True)
class BackendSpec:
    name: str
    directory: Path
    executable: str
    uv_args: tuple[str, ...] = ()


def _find_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if (
            pyproject.exists()
            and (candidate / "configs").exists()
            and (candidate / "python" / "finssim_core").exists()
        ):
            return candidate
    return current


def _backend_spec(workspace_root: Path, backend: str) -> BackendSpec:
    specs = {
        "rl": BackendSpec("rl", workspace_root / "python" / "finssim_rl", "finssim-rl"),
        "marl": BackendSpec("marl", workspace_root / "python" / "finssim_marl", "finssim-marl"),
        "benchmarl": BackendSpec(
            "benchmarl",
            workspace_root / "python" / "finssim_marl",
            "finssim-marl",
        ),
        "marllib": BackendSpec(
            "marllib",
            workspace_root / "python" / "finssim_marl",
            "finssim-marl",
            ("--extra", "marllib"),
        ),
    }
    return specs[backend]


def _kebab(name: str) -> str:
    return name.replace("_", "-")


def _append_override_args(command: list[str], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        flag = f"--{_kebab(str(key))}"
        if value is None:
            continue
        if isinstance(value, bool):
            # TrainArgs uses Optional[bool] here so an omitted YAML value does
            # not override BaseTrainingConfig. Tyro therefore expects an
            # explicit value instead of the usual boolean flag form.
            if key == "keep_eval_snapshots":
                command.extend([flag, str(value)])
                continue
            command.append(flag if value else f"--no-{_kebab(str(key))}")
        elif isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            command.extend([flag, str(value)])
        else:
            raise typer.BadParameter(
                f"trainer.overrides.{key} must be a scalar, bool, list, or tuple"
            )


def _test_exp_name(exp_name: str) -> str:
    return exp_name if exp_name.endswith("_test") else f"{exp_name}_test"


def _effective_test_mode(experiment_config: FinsSimExperimentConfig, cli_test: bool) -> bool:
    return bool(cli_test or experiment_config.trainer.overrides.get("test") is True)


def _resolved_config_payload(
    experiment_config: FinsSimExperimentConfig,
    output_dir: Path,
    exp_name: str | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    experiment_payload = dataclasses.asdict(experiment_config.experiment)
    if exp_name is not None:
        experiment_payload["name"] = exp_name

    trainer_payload = dataclasses.asdict(experiment_config.trainer)
    if overwrite:
        trainer_payload["overrides"]["overwrite"] = True
    if resume:
        trainer_payload["overrides"]["resume"] = True

    payload = {
        "base_config": experiment_config.base_config,
        "experiment": experiment_payload,
        # Keep the role-specific runtime contract in the run artifact.  The
        # backend also receives these values as explicit CLI flags, but this
        # makes the persisted configuration self-contained and auditable.
        "env": {
            "unity": experiment_config.unity_overrides(),
            "train": dataclasses.asdict(experiment_config.train),
            "eval": dataclasses.asdict(experiment_config.eval),
        },
        # Legacy nested payload consumed by existing backend scripts.
        "unity": experiment_config.unity_overrides(),
        "trainer": trainer_payload,
        "artifacts": {
            "output_dir": str(output_dir),
            "logs_dir": str(output_dir / "logs"),
            "checkpoints_dir": str(output_dir / "checkpoints"),
            "tensorboard_dir": str(output_dir / "tensorboard"),
        },
    }
    if "curriculum" in experiment_config.raw:
        payload["curriculum"] = experiment_config.raw["curriculum"]
    if "reward" in experiment_config.raw:
        payload["reward"] = experiment_config.raw["reward"]
    return payload


def _resolve_run_dir(
    workspace_root: Path,
    backend: str,
    experiment_config: FinsSimExperimentConfig,
    exp_name: str | None = None,
) -> Path:
    if experiment_config.experiment.output_dir:
        output_dir = Path(experiment_config.experiment.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = workspace_root / output_dir
        # Explicit output directories do not incorporate exp_name themselves.
        # Keep test artifacts isolated from the corresponding full run.
        if exp_name and exp_name.endswith("_test"):
            output_dir = output_dir.with_name(_test_exp_name(output_dir.name))
        return output_dir

    artifacts_root = default_artifacts_root(workspace_root)
    artifact_backend = "marl" if backend == "benchmarl" else backend
    return artifacts_root / "runs" / artifact_backend / (exp_name or experiment_config.experiment.name)


def _build_backend_command(
    *,
    spec: BackendSpec,
    action: str,
    experiment_config: FinsSimExperimentConfig,
    output_dir: Path,
    resolved_config_path: Path,
    test: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    use_editor: bool = False,
) -> list[str]:
    unity = experiment_config.unity
    role = experiment_config.train if action == "train" else experiment_config.eval
    effective_test = action == "train" and _effective_test_mode(experiment_config, test)
    exp_name = _test_exp_name(experiment_config.experiment.name) if effective_test else experiment_config.experiment.name
    effective_use_editor = bool(use_editor or getattr(unity, "use_editor", False))
    backend_action = action
    if spec.name == "marllib" and action == "train":
        backend_action = "marllib-train"
    elif spec.name == "benchmarl" and action == "train":
        backend_action = "benchmarl-train"
    command = [
        "uv",
        "run",
        *spec.uv_args,
        spec.executable,
        backend_action,
        "--config",
        experiment_config.base_config,
        "--exp-name",
        exp_name,
        "--output-dir",
        str(output_dir),
        "--resolved-config",
        str(resolved_config_path),
    ]

    if experiment_config.has_unity_override("env_base_port"):
        command.extend(["--env-base-port", str(unity.env_base_port)])
    if action == "train" and spec.name == "rl":
        command.extend(["--num-envs", str(experiment_config.train.num_envs)])
        command.extend(["--eval-num-envs", str(experiment_config.eval.num_envs)])
        command.extend(["--time-scale", str(experiment_config.train.time_scale)])
        command.extend(["--eval-time-scale", str(experiment_config.eval.time_scale)])
        command.extend(["--eval-num-episodes", str(experiment_config.eval.num_episodes)])
        command.extend(["--eval-mode", experiment_config.eval.mode])
        command.extend(["--parallel-mode", unity.parallel_mode])
    elif spec.name == "rl":
        command.extend(["--num-envs", str(role.num_envs), "--time-scale", str(role.time_scale), "--parallel-mode", unity.parallel_mode])
        if action == "eval":
            command.extend(["--num-episodes", str(experiment_config.eval.num_episodes)])
    elif spec.name == "marl":
        marl_overrides = dict(experiment_config.trainer.overrides)
        if action == "train":
            command.extend(["--num-envs", str(marl_overrides.get("num_envs", experiment_config.train.num_envs))])
            command.extend(["--num-eval-envs", str(marl_overrides.get("num_eval_envs", experiment_config.eval.num_envs))])
            command.extend(["--time-scale", str(marl_overrides.get("time_scale", experiment_config.train.time_scale))])
            command.extend(["--eval-time-scale", str(marl_overrides.get("eval_time_scale", experiment_config.eval.time_scale))])
            command.extend(["--eval-mode", experiment_config.eval.mode])
        else:
            command.extend(["--num-envs", str(marl_overrides.get("num_eval_envs", role.num_envs))])
            command.extend(["--time-scale", str(marl_overrides.get("eval_time_scale", role.time_scale))])
        command.extend(["--parallel-mode", unity.parallel_mode])
        if action == "eval":
            command.extend(["--num-episodes", str(marl_overrides.get("num_episodes", experiment_config.eval.num_episodes))])
    if (
        ((action == "train" and spec.name in {"rl", "marl"}) or (action == "eval" and spec.name in {"rl", "marl"}))
        and experiment_config.has_unity_override("timeout_wait")
        # Editor evaluation connects to the already-running Unity process.
        # It neither launches nor waits for a Player, so the MARL eval entry
        # point deliberately does not receive a process-start timeout.
        and not (spec.name == "marl" and action == "eval" and effective_use_editor)
    ):
        command.extend(["--timeout-wait", str(unity.timeout_wait)])
    # The RL launcher derives one Unity DR seed per worker from this base seed.
    # ``unity.seed`` is always resolved (falling back to ``experiment.seed``), so
    # it must be forwarded even when the YAML did not explicitly include it.
    # Otherwise every worker starts with BaseEnvironmentConfig.seed=None and the
    # Unity ``-fins-dr-seed`` override is never emitted.
    if spec.name == "rl" and unity.seed is not None:
        command.extend(["--seed", str(unity.seed)])

    if effective_test:
        filtered = []
        skip = False
        test_eval_env_option = "--num-eval-envs" if spec.name == "marl" else "--eval-num-envs"
        for arg in command:
            if skip:
                skip = False
                continue
            if arg in {"--num-envs", test_eval_env_option}:
                skip = True
                continue
            filtered.append(arg)
        command = filtered
        command.extend(["--num-envs", "1", test_eval_env_option, "1"])

    if experiment_config.has_unity_override("env_path") and unity.env_path and not effective_use_editor:
        command.extend(["--env-path", unity.env_path])
    if action == "eval" and spec.name in {"rl", "marl"} and effective_use_editor:
        command.append("--use-editor")
    if experiment_config.has_unity_override("unity_additional_args") and unity.unity_additional_args:
        if spec.name == "rl":
            command.extend(["--unity-additional-args-json", json.dumps(list(unity.unity_additional_args))])
        else:
            command.append("--unity-additional-args")
            command.extend(unity.unity_additional_args)

    if spec.name == "rl" and experiment_config.has_unity_override("port_offset"):
        command.extend(["--port-offset", str(unity.port_offset)])

    if experiment_config.has_unity_override("no_graphics") and not unity.no_graphics:
        command.append("--show-graphics")

    overrides = dict(experiment_config.trainer.overrides)
    if spec.name == "marl":
        if action == "train":
            overrides.pop("num_envs", None)
            overrides.pop("num_eval_envs", None)
            overrides.pop("eval_time_scale", None)
        else:
            overrides.pop("num_eval_envs", None)
            overrides.pop("num_episodes", None)
        overrides.pop("time_scale", None)
    if overwrite:
        # CLI options are explicit user intent and take precedence over YAML.
        overrides["overwrite"] = True
    if resume:
        # Same-run continuation is handled by finssim-rl from checkpoint_dir.
        overrides["resume"] = True
    if action != "train":
        overrides.pop("test", None)
    if effective_test:
        overrides.pop("num_envs", None)
        overrides.pop("num_eval_envs", None)
    _append_override_args(command, overrides)
    if effective_test:
        command = [arg for arg in command if arg != "--no-test"]
        if "--test" not in command:
            command.append("--test")
    return command


def _run_backend(
    backend: str,
    action: str,
    config: Path,
    dry_run: bool,
    test: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    use_editor: bool = False,
) -> None:
    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite are mutually exclusive")
    workspace_root = _find_workspace_root()
    spec = _backend_spec(workspace_root, backend)
    experiment_config = load_experiment_config(config)
    if experiment_config.backend != backend:
        raise typer.BadParameter(
            f"Config backend is '{experiment_config.backend}', but command targets '{backend}'"
        )

    effective_test = action == "train" and _effective_test_mode(experiment_config, test)
    exp_name = _test_exp_name(experiment_config.experiment.name) if effective_test else experiment_config.experiment.name
    run_dir = _resolve_run_dir(workspace_root, backend, experiment_config, exp_name=exp_name)
    resolved_config_path = run_dir / "resolved_config.yaml"
    command = _build_backend_command(
        spec=spec,
        action=action,
        experiment_config=experiment_config,
        output_dir=run_dir,
        resolved_config_path=resolved_config_path,
        test=test,
        overwrite=overwrite,
        resume=resume,
        use_editor=use_editor,
    )

    if dry_run:
        typer.echo(f"Backend cwd: {spec.directory}")
        typer.echo(f"Output dir: {run_dir}")
        typer.echo(f"Command: {shlex.join(command)}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _resolved_config_payload(
        experiment_config,
        run_dir,
        exp_name=exp_name,
        overwrite=overwrite,
        resume=resume,
    )
    write_run_metadata(
        run_dir,
        backend=backend,
        config_path=experiment_config.path,
        run_name=exp_name,
        seed=experiment_config.experiment.seed,
        unity_env_path=experiment_config.unity_overrides().get("env_path"),
        num_envs=experiment_config.train.num_envs if action == "train" else experiment_config.eval.num_envs,
        env_base_port=experiment_config.unity_overrides().get("env_base_port"),
        resolved_config=resolved_config,
        command=command,
        git_root=workspace_root,
    )

    proc = subprocess.Popen(
        command,
        cwd=spec.directory,
        start_new_session=True,
    )
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        typer.echo("Interrupt received, forwarding Ctrl+C to backend and waiting for Unity cleanup...")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            returncode = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = proc.wait()
        raise typer.Exit(code=130)

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


@app.callback()
def main() -> None:
    """Manage FinsSim experiments and runtime backends."""


@app.command()
def info() -> None:
    """Print the current FinsSim workspace layout."""
    workspace_root = _find_workspace_root()
    typer.echo("FinsSim workspace initialized.")
    typer.echo(f"Workspace root: {workspace_root}")
    typer.echo(f"RL backend: {_backend_spec(workspace_root, 'rl').directory}")
    typer.echo(f"MARL backend: {_backend_spec(workspace_root, 'marl').directory}")
    typer.echo(f"BenchMARL backend: {_backend_spec(workspace_root, 'benchmarl').directory}")
    typer.echo(f"MARLlib backend: {_backend_spec(workspace_root, 'marllib').directory}")
    typer.echo(f"Artifacts: {default_artifacts_root(workspace_root)}")


@rl_app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FinsSim RL experiment YAML.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the backend command without launching Unity.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Run backend train in single-environment smoke mode (2,400 steps by default).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite this run's training logs and checkpoints.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume this run from its latest checkpoint without resetting TensorBoard timesteps.",
    ),
) -> None:
    """Launch or preview a single-agent RL training run."""
    _run_backend("rl", "train", config, dry_run, test=test, overwrite=overwrite, resume=resume)


@rl_app.command()
def eval(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FinsSim RL evaluation YAML.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the backend command without launching Unity.",
    ),
    use_editor: bool = typer.Option(
        False,
        "--use-editor",
        help="Connect to a running Unity Editor Play session instead of launching a Unity binary.",
    ),
) -> None:
    """Launch or preview a single-agent RL evaluation run."""
    _run_backend("rl", "eval", config, dry_run, use_editor=use_editor)


@marl_app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FinsSim MARL experiment YAML.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the backend command without launching Unity.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Run backend train in single-environment smoke mode (2,400 steps by default).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite this run's training logs and checkpoints.",
    ),
) -> None:
    """Launch or preview a multi-agent RL training run."""
    _run_backend("marl", "train", config, dry_run, test=test, overwrite=overwrite)


@marl_app.command()
def eval(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FinsSim MARL evaluation YAML.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the backend command without launching Unity.",
    ),
    use_editor: bool = typer.Option(
        False,
        "--use-editor",
        help="Connect to a running Unity Editor Play session instead of launching a Unity binary.",
    ),
) -> None:
    """Launch or preview a multi-agent RL evaluation run."""
    _run_backend("marl", "eval", config, dry_run, use_editor=use_editor)


benchmarl_app = typer.Typer(help="BenchMARL/TorchRL benchmark backend commands.")


@benchmarl_app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FinsSim BenchMARL experiment YAML.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the backend command without launching Unity.",
    ),
) -> None:
    """Launch or preview a BenchMARL/TorchRL training setup."""
    _run_backend("benchmarl", "train", config, dry_run)


marllib_app = typer.Typer(help="MARLlib benchmark backend commands.")
isaaclab_app = typer.Typer(help="Isaac Lab RSL-RL and SKRL training commands.")


@marllib_app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FinsSim MARLlib benchmark YAML.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the backend command without launching Unity.",
    ),
) -> None:
    """Launch or preview a MARLlib benchmark training run."""
    _run_backend("marllib", "train", config, dry_run)


@isaaclab_app.command(name="train")
def isaaclab_train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, file_okay=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate YAML and print the IsaacLab command."),
) -> None:
    """Train an IsaacLab task from a FinsSim IsaacLab experiment YAML."""
    launch_isaaclab(config, workspace_root=_find_workspace_root(), action="train", dry_run=dry_run)


@isaaclab_app.command(name="play")
def isaaclab_play(
    config: Path = typer.Option(..., "--config", "-c", exists=True, file_okay=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate YAML and print the IsaacLab command."),
) -> None:
    """Replay the YAML-selected IsaacLab checkpoint."""
    launch_isaaclab(config, workspace_root=_find_workspace_root(), action="play", dry_run=dry_run)


app.add_typer(rl_app, name="rl")
app.add_typer(marl_app, name="marl")
app.add_typer(benchmarl_app, name="benchmarl")
app.add_typer(marllib_app, name="marllib")
app.add_typer(isaaclab_app, name="isaaclab")
