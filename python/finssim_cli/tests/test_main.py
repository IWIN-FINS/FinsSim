from __future__ import annotations

from pathlib import Path

from finssim_cli.main import (
    BackendSpec,
    _build_backend_command,
    _resolved_config_payload,
    _resolve_run_dir,
)
from finssim_core.config_loader import load_experiment_config


def _write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_build_backend_command_only_forwards_explicit_unity_overrides(tmp_path):
    config_path = _write_config(
        tmp_path / "marl.yaml",
        """
base_config: chasing_3_chase_1_end_to_end
experiment:
  name: smoke
unity:
  env_base_port: 3100
trainer:
  backend: marl
  overrides:
    overwrite: true
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marl", tmp_path, "finssim-marl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert command == [
        "uv",
        "run",
        "finssim-marl",
        "train",
        "--config",
        "chasing_3_chase_1_end_to_end",
        "--exp-name",
        "smoke",
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--resolved-config",
        str(tmp_path / "artifacts" / "resolved_config.yaml"),
        "--env-base-port",
        "3100",
        "--num-envs",
        "1",
        "--num-eval-envs",
        "1",
        "--time-scale",
        "10.0",
        "--eval-time-scale",
        "10.0",
        "--eval-mode",
        "asynchronous",
        "--parallel-mode",
        "multi_area",
        "--overwrite",
    ]


def test_build_backend_command_preserves_explicit_zero_and_false_overrides(tmp_path):
    config_path = _write_config(
        tmp_path / "rl.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity:
  port_offset: 0
  no_graphics: false
  timeout_wait: 180
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--port-offset" in command
    assert "0" in command
    assert "--show-graphics" in command
    assert "--timeout-wait" in command
    assert "180" in command
    # The role configuration is always explicit at the backend boundary,
    # including inherited defaults from the normalized env.train block.
    assert command[command.index("--num-envs") + 1] == "1"
    assert command[command.index("--time-scale") + 1] == "10.0"


def test_build_rl_backend_command_forwards_eval_snapshot_retention(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_keep_eval_snapshots.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity: {}
trainer:
  backend: rl
  overrides:
    keep_eval_snapshots: true
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    index = command.index("--keep-eval-snapshots")
    assert command[index + 1] == "True"


def test_build_rl_backend_command_forwards_explicit_unity_seed(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_seed.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity:
  seed: 77
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "77"


def test_build_rl_backend_command_forwards_effective_experiment_seed(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_experiment_seed.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
  seed: 91
unity: {}
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "91"


def test_build_marl_backend_command_forwards_timeout_wait(tmp_path):
    config_path = _write_config(
        tmp_path / "marl_timeout.yaml",
        """
base_config: chasing_3_chase_1_end_to_end
experiment:
  name: smoke
unity:
  env_base_port: 3100
  timeout_wait: 240
trainer:
  backend: marl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marl", tmp_path, "finssim-marl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--timeout-wait" in command
    assert command[command.index("--timeout-wait") + 1] == "240"


def test_build_backend_command_forwards_test_override(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_test.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity:
  num_envs: 8
trainer:
  backend: rl
  overrides:
    test: true
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--num-envs" in command
    assert command[command.index("--num-envs") + 1] == "1"
    assert "--eval-num-envs" in command
    assert command[command.index("--eval-num-envs") + 1] == "1"
    assert command[command.index("--exp-name") + 1] == "smoke_test"
    assert "--test" in command


def test_build_rl_backend_command_forwards_explicit_num_eval_envs(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_num_eval.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity:
  num_envs: 8
  num_eval_envs: 3
  num_eval_areas: 5
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--num-envs" in command
    assert command[command.index("--num-envs") + 1] == "8"
    assert "--eval-num-envs" in command
    # Legacy num_eval_areas is normalized to the unified env.eval.num_envs.
    assert command[command.index("--eval-num-envs") + 1] == "5"
    assert "--num-eval-areas" not in command


def test_build_rl_eval_command_forwards_timeout_wait(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_eval.yaml",
        """
base_config: hierarchy_chase_1chase1
experiment:
  name: smoke_eval
unity:
  env_path: ../../artifacts/unity_builds/rl/linux/FinsROV/1Chase1_visual/1Chase1.x86_64
  env_base_port: 16125
  timeout_wait: 600
  no_graphics: false
trainer:
  backend: rl
  overrides:
    checkpoint_path: ../../artifacts/runs/rl/1chase1/hierarchy_chase/checkpoints/best_model.zip
    num_episodes: 1
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="eval",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--env-base-port" in command
    assert "--show-graphics" in command
    assert "--timeout-wait" in command
    assert command[command.index("--timeout-wait") + 1] == "600"
    assert "--checkpoint-path" in command
    assert command[command.index("--checkpoint-path") + 1].endswith(
        "/hierarchy_chase/checkpoints/best_model.zip"
    )


def test_build_rl_eval_command_forwards_unity_additional_args_as_json(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_eval_args.yaml",
        """
base_config: hybrid_pid_pose_v3
experiment:
  name: smoke_eval
unity:
  env_path: /tmp/1Chase1.x86_64
  unity_additional_args:
    - -fins-1chase1-mode
    - direct14
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="eval",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--unity-additional-args" not in command
    assert "--unity-additional-args-json" in command
    payload = command[command.index("--unity-additional-args-json") + 1]
    assert payload == '["-fins-1chase1-mode", "direct14"]'


def test_build_rl_eval_command_forwards_use_editor(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_eval_editor.yaml",
        """
base_config: hybrid_pid_pose_v3
experiment:
  name: smoke_editor_eval
unity:
  use_editor: true
  env_base_port: 5004
  no_graphics: false
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="eval",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert "--use-editor" in command
    assert "--env-path" not in command


def test_build_rl_eval_command_cli_use_editor_overrides_env_path(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_eval_cli_editor.yaml",
        """
base_config: hybrid_pid_pose_v3
experiment:
  name: smoke_cli_editor_eval
unity:
  env_path: /tmp/1Chase1.x86_64
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="eval",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
        use_editor=True,
    )

    assert "--use-editor" in command
    assert "--env-path" not in command


def test_build_marl_eval_command_forwards_use_editor(tmp_path):
    config_path = _write_config(
        tmp_path / "marl_eval_editor.yaml",
        """
base_config: chasing_3_chase_1_end_to_end
experiment:
  name: smoke_editor_eval
unity:
  env_path: /tmp/3Chase1.x86_64
  timeout_wait: 600
trainer:
  backend: marl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marl", tmp_path, "finssim-marl"),
        action="eval",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
        use_editor=True,
    )

    assert "--parallel-mode" in command
    assert command[command.index("--parallel-mode") + 1] == "multi_area"
    assert "--use-editor" in command
    assert "--env-path" not in command
    assert "--timeout-wait" not in command


def test_build_backend_command_cli_test_overrides_yaml_no_test(tmp_path):
    config_path = _write_config(
        tmp_path / "marl_no_test.yaml",
        """
base_config: chasing_3_chase_1
experiment:
  name: smoke
unity:
  num_envs: 8
trainer:
  backend: marl
  overrides:
    test: false
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marl", tmp_path, "finssim-marl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
        test=True,
    )

    assert "--no-test" not in command
    assert command[command.index("--exp-name") + 1] == "smoke_test"
    assert command[-1] == "--test"


def test_build_backend_command_does_not_duplicate_test_suffix(tmp_path):
    config_path = _write_config(
        tmp_path / "marl_test.yaml",
        """
base_config: chasing_3_chase_1
experiment:
  name: smoke_test
unity: {}
trainer:
  backend: marl
  overrides:
    test: true
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marl", tmp_path, "finssim-marl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert command[command.index("--exp-name") + 1] == "smoke_test"


def test_resolve_run_dir_can_use_test_exp_name(tmp_path):
    config_path = _write_config(
        tmp_path / "rl.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity: {}
trainer:
  backend: rl
  overrides:
    test: true
""",
    )
    config = load_experiment_config(config_path)

    run_dir = _resolve_run_dir(tmp_path, "rl", config, exp_name="smoke_test")

    assert run_dir == tmp_path / "artifacts" / "runs" / "rl" / "smoke_test"


def test_resolve_run_dir_adds_test_suffix_to_explicit_output_dir(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_test_output.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
  output_dir: artifacts/runs/rl/custom/smoke_run
unity: {}
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    run_dir = _resolve_run_dir(tmp_path, "rl", config, exp_name="smoke_test")

    assert run_dir == tmp_path / "artifacts" / "runs" / "rl" / "custom" / "smoke_run_test"


def test_build_backend_command_cli_overwrite_overrides_yaml(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_overwrite.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity: {}
trainer:
  backend: rl
  overrides:
    overwrite: false
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
        overwrite=True,
    )

    assert "--overwrite" in command
    assert "--no-overwrite" not in command


def test_build_rl_backend_command_cli_resume_overrides_yaml(tmp_path):
    config_path = _write_config(
        tmp_path / "rl_resume.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity: {}
trainer:
  backend: rl
  overrides:
    resume: false
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("rl", tmp_path, "finssim-rl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
        resume=True,
    )

    assert "--resume" in command
    assert "--no-resume" not in command


def test_build_marl_backend_command_cli_overwrite_overrides_yaml(tmp_path):
    config_path = _write_config(
        tmp_path / "marl_overwrite.yaml",
        """
base_config: chasing_3_chase_1
experiment:
  name: smoke
unity: {}
trainer:
  backend: marl
  overrides:
    overwrite: false
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marl", tmp_path, "finssim-marl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
        overwrite=True,
    )

    assert "--overwrite" in command
    assert "--no-overwrite" not in command


def test_build_marllib_backend_command_uses_extra_and_subcommand(tmp_path):
    config_path = _write_config(
        tmp_path / "marllib.yaml",
        """
base_config: unity_3chase1
experiment:
  name: smoke
unity:
  env_base_port: 7100
trainer:
  backend: marllib
  overrides:
    algorithm: mappo
    stop_iters: 1
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("marllib", tmp_path, "finssim-marl", ("--extra", "marllib")),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert command[:6] == ["uv", "run", "--extra", "marllib", "finssim-marl", "marllib-train"]
    assert "--config" in command
    assert "unity_3chase1" in command
    assert "--algorithm" in command
    assert "mappo" in command


def test_build_benchmarl_backend_command_uses_subcommand(tmp_path):
    config_path = _write_config(
        tmp_path / "benchmarl.yaml",
        """
base_config: chasing_3chase1
experiment:
  name: smoke
unity:
  env_base_port: 8100
  num_envs: 1
trainer:
  backend: benchmarl
  overrides:
    dry_run: true
""",
    )
    config = load_experiment_config(config_path)

    command = _build_backend_command(
        spec=BackendSpec("benchmarl", tmp_path, "finssim-marl"),
        action="train",
        experiment_config=config,
        output_dir=tmp_path / "artifacts",
        resolved_config_path=tmp_path / "artifacts" / "resolved_config.yaml",
    )

    assert command[:4] == ["uv", "run", "finssim-marl", "benchmarl-train"]
    assert "--config" in command
    assert "chasing_3chase1" in command
    assert "--num-envs" not in command
    assert "--dry-run" in command


def test_resolve_run_dir_uses_relative_experiment_output_dir(tmp_path):
    config_path = _write_config(
        tmp_path / "rl.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
  output_dir: artifacts/runs/rl/new_reward/smoke
unity: {}
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    run_dir = _resolve_run_dir(tmp_path, "rl", config)

    assert run_dir == tmp_path / "artifacts" / "runs" / "rl" / "new_reward" / "smoke"


def test_resolve_run_dir_uses_absolute_experiment_output_dir(tmp_path):
    output_dir = tmp_path / "custom" / "run"
    config_path = _write_config(
        tmp_path / "rl.yaml",
        f"""
base_config: control_for_pose
experiment:
  name: smoke
  output_dir: {output_dir}
unity: {{}}
trainer:
  backend: rl
  overrides: {{}}
""",
    )
    config = load_experiment_config(config_path)

    run_dir = _resolve_run_dir(tmp_path / "workspace", "rl", config)

    assert run_dir == output_dir


def test_resolve_run_dir_places_benchmarl_under_marl_artifacts(tmp_path):
    config_path = _write_config(
        tmp_path / "benchmarl.yaml",
        """
base_config: chasing_3chase1
experiment:
  name: smoke
unity: {}
trainer:
  backend: benchmarl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    run_dir = _resolve_run_dir(tmp_path, "benchmarl", config)

    assert run_dir == tmp_path / "artifacts" / "runs" / "marl" / "smoke"


def test_resolved_config_payload_only_records_explicit_unity_overrides(tmp_path):
    config_path = _write_config(
        tmp_path / "rl.yaml",
        """
base_config: control_for_pose
experiment:
  name: smoke
unity:
  env_path: /tmp/env.x86_64
  no_graphics: false
trainer:
  backend: rl
  overrides: {}
""",
    )
    config = load_experiment_config(config_path)

    payload = _resolved_config_payload(config, tmp_path / "artifacts")

    assert payload["unity"] == {
        "env_path": "/tmp/env.x86_64",
        "no_graphics": False,
    }
    assert payload["env"]["train"] == {
        "num_envs": 1,
        "time_scale": 10.0,
        "no_graphics": None,
        "timeout_wait": None,
        "mode": "asynchronous",
        "num_episodes": 5,
    }


def test_resolved_config_payload_preserves_curriculum_and_environment_parameters(tmp_path):
    config_path = _write_config(
        tmp_path / "marl_curriculum.yaml",
        """
base_config: chasing_3_chase_1_end_to_end
experiment:
  name: smoke
unity:
  environment_parameters:
    finsim_3c1.lesson_id: 0
trainer:
  backend: marl
  overrides: {}
curriculum:
  enabled: true
  lessons:
    - name: l0
      start_step: 0
      parameters:
        finsim_3c1.lesson_id: 1
""",
    )
    config = load_experiment_config(config_path)

    payload = _resolved_config_payload(config, tmp_path / "artifacts")

    assert payload["unity"] == {
        "environment_parameters": {
            "finsim_3c1.lesson_id": 0.0,
        }
    }
    assert payload["env"]["unity"] == payload["unity"]
    assert payload["curriculum"]["enabled"] is True
    assert payload["curriculum"]["lessons"][0]["parameters"]["finsim_3c1.lesson_id"] == 1
