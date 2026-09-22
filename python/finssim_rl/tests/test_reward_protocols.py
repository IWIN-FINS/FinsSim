from __future__ import annotations

import pytest

from finssim_rl.training.reward_protocols import (
    compile_reward_config,
    resolved_unity_environment_parameters,
)


def test_compile_distance_plus_subgoal_reward() -> None:
    compiled = compile_reward_config(
        {
            "protocol": "one_chase_one_reward_v1",
            "mode": "distance_plus_subgoal",
            "parameters": {
                "distance_scale": 0.12,
                "subgoal_to_prey_scale": 0.34,
            },
        }
    )

    assert compiled is not None
    assert compiled.protocol == "one_chase_one_reward_v1"
    assert compiled.version == 1
    assert compiled.mode_name == "distance_plus_subgoal"
    assert compiled.mode_id == 2
    assert compiled.environment_parameters["finsim_1c1.reward.protocol_version"] == 1.0
    assert compiled.environment_parameters["finsim_1c1.reward.mode"] == 2.0
    assert compiled.environment_parameters["finsim_1c1.reward.distance_scale"] == 0.12
    assert compiled.environment_parameters["finsim_1c1.reward.subgoal_to_prey_scale"] == 0.34


def test_unknown_reward_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown reward.mode"):
        compile_reward_config(
            {
                "protocol": "one_chase_one_reward_v1",
                "mode": "not_a_reward",
            }
        )


def test_unknown_reward_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown reward.parameters keys"):
        compile_reward_config(
            {
                "protocol": "one_chase_one_reward_v1",
                "mode": "distance_only",
                "parameters": {
                    "typo_scale": 1.0,
                },
            }
        )


def test_resolved_unity_environment_parameters_supports_platform_snapshot() -> None:
    parameters = resolved_unity_environment_parameters(
        {
            "unity": {
                "environment_parameters": {
                    "finsim_hold.reward.mode": 1.0,
                }
            }
        }
    )

    assert parameters == {"finsim_hold.reward.mode": 1.0}


def test_resolved_unity_environment_parameters_uses_explicit_platform_values() -> None:
    parameters = resolved_unity_environment_parameters(
        {
            "final_config": {
                "env_config": {
                    "environment_parameters": {
                        "finsim_hold.reward.mode": 0.0,
                    }
                }
            },
            "unity": {
                "environment_parameters": {
                    "finsim_hold.reward.mode": 1.0,
                }
            },
        }
    )

    assert parameters == {"finsim_hold.reward.mode": 1.0}
