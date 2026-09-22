from dataclasses import replace

import pytest

from finssim_rl.models import _unity_additional_args, _unity_domain_randomization_args
from finssim_rl.training.config import BaseEnvironmentConfig


def test_unity_domain_randomization_seed_is_unique_per_worker_and_reproducible():
    config = BaseEnvironmentConfig(seed=12345)

    first_pass = [_unity_domain_randomization_args(config, rank) for rank in range(32)]
    second_pass = [_unity_domain_randomization_args(config, rank) for rank in range(32)]

    assert first_pass == second_pass
    assert all(args[0] == "-fins-dr-seed" for args in first_pass)
    assert len({args[1] for args in first_pass}) == 32
    assert first_pass[0] == ["-fins-dr-seed", "12345"]
    assert first_pass[1] == ["-fins-dr-seed", "1012348"]


def test_unity_domain_randomization_seed_is_not_emitted_without_a_base_seed():
    assert _unity_domain_randomization_args(BaseEnvironmentConfig(), 0) == []


@pytest.mark.parametrize("configured_args", [["-fins-dr-seed", "777"], ["-fins-dr-seed=777"]])
def test_unity_additional_args_rejects_manual_domain_randomization_seed(configured_args):
    config = replace(BaseEnvironmentConfig(seed=12345), unity_additional_args=configured_args)

    with pytest.raises(ValueError, match="managed by the RL launcher"):
        _unity_additional_args(config, 0)


def test_unity_additional_args_preserves_other_player_arguments_after_dr_seed():
    config = replace(
        BaseEnvironmentConfig(seed=12345),
        unity_additional_args=["-fins-dr-mode", "train"],
    )

    assert _unity_additional_args(config, 2) == [
        "-fins-dr-seed",
        "2012351",
        "-fins-dr-mode",
        "train",
    ]
