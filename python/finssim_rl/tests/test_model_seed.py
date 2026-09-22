from types import SimpleNamespace

from finssim_rl.models import ppo_control, ppo_control_v2
from finssim_rl.models.ppo_control import PPOConfig
from finssim_rl.models.ppo_control_v2 import PPOV2Config


class _CapturedModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_standard_ppo_forwards_cli_seed_to_sb3(monkeypatch):
    monkeypatch.setattr(ppo_control, "PPO", _CapturedModel)

    model = PPOConfig(name="seed-test").create_model(
        env=object(),
        args=SimpleNamespace(device="cpu", tensorboard_dir=None, seed=123),
    )

    assert model.kwargs["seed"] == 123


def test_virtual_wrench_ppo_forwards_cli_seed_to_sb3(monkeypatch):
    monkeypatch.setattr(ppo_control_v2, "PPOVirtualControlModel", _CapturedModel)

    model = PPOV2Config(name="seed-test").create_model(
        env=object(),
        args=SimpleNamespace(device="cpu", tensorboard_dir=None, seed=456),
    )

    assert model.kwargs["seed"] == 456
